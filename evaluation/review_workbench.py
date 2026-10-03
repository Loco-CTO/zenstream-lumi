"""Loopback-only blind annotation workbench for development cases.

This tool accepts development cases only, refuses interactive tool fixtures whose
expected path could be exposed, and never returns gold labels or review history
to the browser. Review records are appended to a caller-selected controlled file.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import secrets
import sys
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.review_records import (  # noqa: E402
    ReviewRecordError,
    read_review_records,
    validate_case_review_records,
)
from evaluation.scorer import EvaluationInputError, _validate_case  # noqa: E402
from provenance.validate import evaluation_case_sha256  # noqa: E402


REVIEWER_ID_PATTERN = re.compile(r"^rev-[a-z0-9]+(?:-[a-z0-9]+)*$")
LANGUAGE_QUALIFICATIONS = {
    "ja": {"native_japanese", "fluent_japanese"},
    "en_ja": {"fluent_bilingual"},
}


class WorkbenchError(ValueError):
    """A safe-to-display workbench input or review error."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not allowed")
        result[key] = value
    return result


def _read_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    case = json.loads(
                        line,
                        parse_constant=_reject_constant,
                        object_pairs_hook=_reject_duplicate_keys,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise WorkbenchError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(case, dict):
                    raise WorkbenchError(f"{path}:{line_number}: each case must be an object")
                try:
                    _validate_case(case, line_number)
                except EvaluationInputError as exc:
                    raise WorkbenchError(f"{path}: {exc}") from exc
                case_id = case.get("case_id")
                if not isinstance(case_id, str) or not case_id.strip():
                    raise WorkbenchError(f"{path}:{line_number}: case_id must be a nonempty string")
                if case_id in seen:
                    raise WorkbenchError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
                seen.add(case_id)
                if case.get("schema_version") != 4:
                    raise WorkbenchError(f"{path}:{line_number}: only case schema v4 is supported")
                if case.get("split") != "development":
                    raise WorkbenchError("the workbench accepts development cases only")
                if case.get("review_status") != "draft":
                    raise WorkbenchError(
                        f"case {case_id!r} is not a draft; prepare a draft-only review inventory"
                    )
                if case.get("tool_scenario") is not None:
                    raise WorkbenchError(
                        "interactive tool scenarios are not supported because their expected fixture path "
                        "must stay hidden from independent reviewers"
                    )
                if not isinstance(case.get("language"), str) or case["language"] not in {
                    "en", "ja", "en_ja",
                }:
                    raise WorkbenchError(f"case {case_id!r} has an unsupported language")
                evaluation_case_sha256(case)
                cases.append(case)
    except UnicodeDecodeError as exc:
        raise WorkbenchError(f"{path}: case inventory is not UTF-8") from exc
    if not cases:
        raise WorkbenchError("the development case inventory is empty")
    return cases


def _active_records(case: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_hash = evaluation_case_sha256(case)
    superseded = {
        record_id
        for record in records
        for record_id in record.get("supersedes_record_ids", [])
    }
    return [
        record for record in records
        if record.get("case_id") == case["case_id"]
        and record.get("case_sha256") == case_hash
        and record.get("review_record_id") not in superseded
    ]


def _validate_ledger(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    """Validate ledger records against current hashes without promoting draft cases."""
    for index, record in enumerate(records, start=1):
        if (
            not isinstance(record.get("review_record_id"), str)
            or not isinstance(record.get("case_id"), str)
            or not isinstance(record.get("case_sha256"), str)
            or not isinstance(record.get("reviewer_id"), str)
            or not isinstance(record.get("supersedes_record_ids"), list)
            or not all(isinstance(item, str) for item in record["supersedes_record_ids"])
        ):
            raise ReviewRecordError(f"review record {index}: malformed identity or supersession fields")
    copies = copy.deepcopy(cases)
    for case in copies:
        active = _active_records(case, records)
        case["review_status"] = "draft"
        case["review"]["review_record_ids"] = [
            record["review_record_id"] for record in active
        ]
        case["review"]["reviewer_ids"] = sorted({
            record["reviewer_id"] for record in active
        })
    validate_case_review_records(copies, records)


def _safe_output_path(path: Path, cases_path: Path) -> Path:
    output = path.expanduser().resolve()
    cases = cases_path.expanduser().resolve()
    root = REPOSITORY_ROOT.resolve()
    if output == cases:
        raise WorkbenchError("review records must be written to a separate file")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise WorkbenchError("review records must be stored outside the Git repository")
    if output.suffix.lower() != ".jsonl":
        raise WorkbenchError("the review output path must end in .jsonl")
    if output.exists() and not output.is_file():
        raise WorkbenchError("the review output path is not a regular file")
    return output


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkbenchError(f"request body is not valid strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkbenchError("request body must be a JSON object")
    return value


class ReviewManager:
    def __init__(
        self,
        cases: list[dict[str, Any]],
        output_path: Path,
        reviewer_id: str,
        role: str,
    ) -> None:
        self.cases = cases
        self.output_path = output_path
        self.reviewer_id = reviewer_id
        self.role = role
        self.lock = threading.Lock()
        try:
            self.records = read_review_records(output_path) if output_path.exists() else []
            _validate_ledger(self.cases, self.records)
        except ReviewRecordError as exc:
            raise WorkbenchError(f"review ledger is invalid: {exc}") from exc
        self.assigned = [
            index for index, case in enumerate(self.cases)
            if role == "semantic" or case["language"] in LANGUAGE_QUALIFICATIONS
        ]
        if role == "language" and not self.assigned:
            raise WorkbenchError("this inventory has no Japanese or code-switched cases to review")

    def safe_cases(self) -> list[dict[str, Any]]:
        """Expose only prompt-visible case inputs, excluding IDs, labels and provenance."""
        return [
            {
                "index": index,
                "language": self.cases[index]["language"],
                "turns": self.cases[index]["turns"],
                "trusted_context": self.cases[index]["trusted_context"],
            }
            for index in self.assigned
        ]

    def completed_indices(self) -> list[int]:
        wanted_type = "independent_annotation" if self.role == "semantic" else "language_review"
        completed: list[int] = []
        for index in self.assigned:
            case = self.cases[index]
            if any(
                record.get("reviewer_id") == self.reviewer_id
                and record.get("record_type") == wanted_type
                for record in _active_records(case, self.records)
            ):
                completed.append(index)
        return completed

    def _make_record(self, case: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        common: dict[str, Any] = {
            "schema_version": 1,
            "review_record_id": f"review-{uuid.uuid4()}",
            "case_id": case["case_id"],
            "case_sha256": evaluation_case_sha256(case),
            "reviewer_id": self.reviewer_id,
            "reviewed_at": now,
            "supersedes_record_ids": [],
        }
        if self.role == "semantic":
            if set(payload) != {"index", "proposed_gold"}:
                raise WorkbenchError("semantic review submission has unexpected fields")
            proposed = payload["proposed_gold"]
            if not isinstance(proposed, dict):
                raise WorkbenchError("proposed_gold must be an object")
            return {
                **common,
                "record_type": "independent_annotation",
                "proposed_gold": proposed,
            }

        if set(payload) != {"index", "language_review"}:
            raise WorkbenchError("language review submission has unexpected fields")
        review = payload["language_review"]
        if not isinstance(review, dict) or set(review) != {
            "qualification", "naturalness_status", "meaning_preservation_status",
        }:
            raise WorkbenchError("language review fields are incomplete or unexpected")
        case_language = case["language"]
        if case_language not in LANGUAGE_QUALIFICATIONS:
            raise WorkbenchError("language review applies only to Japanese or code-switched cases")
        if (
            not isinstance(review["qualification"], str)
            or review["qualification"] not in LANGUAGE_QUALIFICATIONS[case_language]
        ):
            raise WorkbenchError("reviewer qualification does not meet this case language requirement")
        if (
            not isinstance(review["naturalness_status"], str)
            or review["naturalness_status"] not in {"approved", "needs_revision"}
        ):
            raise WorkbenchError("naturalness status must be approved or needs_revision")
        if (
            not isinstance(review["meaning_preservation_status"], str)
            or review["meaning_preservation_status"] not in {"approved", "needs_revision"}
        ):
            raise WorkbenchError("meaning status must be approved or needs_revision")
        return {
            **common,
            "record_type": "language_review",
            "language": case_language,
            **review,
        }

    def submit(self, payload: dict[str, Any]) -> dict[str, int]:
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index not in self.assigned:
            raise WorkbenchError("case index is not assigned in this review session")
        case = self.cases[index]
        record = self._make_record(case, payload)
        with self.lock:
            active = _active_records(case, self.records)
            if any(
                item.get("reviewer_id") == self.reviewer_id
                and item.get("record_type") == record["record_type"]
                for item in active
            ):
                raise WorkbenchError(
                    "this reviewer has already submitted this case; corrections need an explicit superseding record"
                )
            if any(
                item.get("reviewer_id") == self.reviewer_id
                and item.get("record_type") in {"independent_annotation", "adjudication", "language_review"}
                for item in active
            ):
                raise WorkbenchError(
                    "the same reviewer cannot provide both semantic and language review for one case"
                )
            candidate_records = [*self.records, record]
            _validate_ledger(self.cases, candidate_records)
            serialized = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode("utf-8")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.output_path,
                os.O_CREAT | os.O_APPEND | os.O_WRONLY,
                0o600,
            )
            try:
                remaining = memoryview(serialized)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("short write while appending review record")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.records.append(record)
        return {
            "completed": len(self.completed_indices()),
            "total": len(self.assigned),
        }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lumi development review</title>
  <style nonce="__NONCE__">
    :root { color-scheme: light dark; font: 16px/1.55 system-ui, sans-serif; }
    body { margin: 0; background: #111827; color: #f3f4f6; }
    main { width: min(900px, calc(100% - 32px)); margin: 28px auto 60px; }
    h1 { margin-bottom: 4px; font-size: 1.7rem; }
    .muted { color: #b4bdcc; }
    .notice, .panel { border: 1px solid #374151; border-radius: 12px; padding: 16px; margin: 16px 0; background: #1f2937; }
    .notice { border-color: #806b35; background: #302a1c; }
    .turn { padding: 12px; margin: 9px 0; border-left: 3px solid #6b7280; background: #111827; white-space: pre-wrap; overflow-wrap: anywhere; }
    .turn.user { border-color: #60a5fa; } .turn.assistant { border-color: #a78bfa; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 280px; overflow: auto; background: #111827; padding: 12px; border-radius: 8px; }
    label { display: block; margin: 12px 0 5px; font-weight: 600; }
    input, select, textarea { box-sizing: border-box; width: 100%; padding: 10px; border: 1px solid #4b5563; border-radius: 8px; background: #111827; color: inherit; font: inherit; }
    textarea { min-height: 115px; font-family: ui-monospace, monospace; }
    .inline { display: flex; gap: 12px; align-items: center; }
    .inline > * { flex: 1; }
    .check { display: flex; gap: 10px; align-items: center; font-weight: 400; }
    .check input { width: auto; }
    .toolbar { display: flex; gap: 10px; margin-top: 18px; }
    button { border: 0; border-radius: 8px; padding: 11px 16px; font: inherit; font-weight: 700; cursor: pointer; }
    button.primary { background: #60a5fa; color: #111827; } button.secondary { background: #374151; color: #f9fafb; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    #error { color: #fca5a5; min-height: 1.5em; }
    .badge { display: inline-block; padding: 2px 9px; border-radius: 999px; background: #374151; color: #d1d5db; font-size: .85rem; }
    @media (max-width: 620px) { .inline, .toolbar { flex-direction: column; align-items: stretch; } }
  </style>
</head>
<body>
<main>
  <h1>Lumi development case review</h1>
  <div id="session" class="muted">Loading local review session…</div>
  <section class="notice">
    <strong>Independent review</strong>
    <div>Review the shown conversation and trusted context on its own. Gold labels, case categories, provenance, and other reviewers’ decisions are hidden. Do not discuss cases with other reviewers until your independent decisions are saved.</div>
    <div class="muted">This local tool records your judgments only. It does not approve data rights, privacy, contamination, or case readiness. Reviewer qualification is self-reported.</div>
  </section>
  <div id="error" role="alert"></div>
  <section id="case" class="panel" aria-live="polite"></section>
</main>
<script nonce="__NONCE__">
(() => {
  const caseBox = document.getElementById("case");
  const errorBox = document.getElementById("error");
  const sessionBox = document.getElementById("session");
  let session = null, cases = [], completed = new Set(), skipped = new Set(), current = null;
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };
  const field = (title, control) => {
    const wrap = element("div");
    wrap.append(element("label", title));
    wrap.append(control);
    return wrap;
  };
  const select = (items) => {
    const node = element("select");
    for (const [value, label] of items) {
      const option = element("option", label);
      option.value = value;
      node.append(option);
    }
    return node;
  };
  async function getJson(url, options) {
    const response = await fetch(url, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The local review request failed.");
    return result;
  }
  function remaining() {
    return cases.filter(item => !completed.has(item.index) && !skipped.has(item.index));
  }
  function renderInputs(item) {
    caseBox.replaceChildren();
    const header = element("div");
    const ordinal = cases.findIndex(candidate => candidate.index === item.index) + 1;
    header.append(element("strong", "Development case " + ordinal + " of " + session.total));
    header.append(document.createTextNode(" "));
    header.append(element("span", item.language, "badge"));
    caseBox.append(header);
    caseBox.append(element("h2", "Conversation"));
    for (const turn of item.turns) {
      const block = element("div", turn.role + ": " + turn.text, "turn " + turn.role);
      caseBox.append(block);
    }
    caseBox.append(element("h2", "Trusted context"));
    const context = element("pre", item.trusted_context === null
      ? "No trusted context was supplied."
      : JSON.stringify(item.trusted_context, null, 2));
    caseBox.append(context);
    if (session.role === "semantic") renderSemanticForm();
    else renderLanguageForm(item);
  }
  function renderSemanticForm() {
    const form = element("div", null, "panel");
    const decision = select([
      ["act", "Act: a capability is requested now"],
      ["clarify", "Clarify: intent is present but underspecified"],
      ["respond", "Respond: answer without current ZenStream state"],
      ["no_action", "No action: mention, rejection, hypothetical, or discussion"],
    ]);
    const action = element("input"); action.placeholder = "Reviewed capability ID, if act";
    const args = element("textarea"); args.value = "{}";
    const includeResponse = element("input"); includeResponse.type = "checkbox";
    const responseReq = select([["required","Required"],["optional","Optional"],["forbidden","Forbidden"]]);
    const responseLang = select([["same_as_case","Same as case"],["en","English"],["ja","Japanese"],["en_ja","English/Japanese"],["any","Any"]]);
    const responseFields = element("div"); responseFields.hidden = true;
    responseFields.append(element("label","Response requirement"), responseReq);
    responseFields.append(element("label","Response language target"), responseLang);
    const presentation = select([
      ["","Not annotated"],["none","None"],["text","Text"],["media_results","Media results"],
      ["media_details","Media details"],["playback_handoff","Playback handoff"],["confirmation","Confirmation"],
    ]);
    const updateDecision = () => {
      const acting = decision.value === "act";
      action.disabled = !acting; args.disabled = !acting;
      if (!acting) { action.value = ""; args.value = "{}"; }
    };
    decision.addEventListener("change", updateDecision);
    includeResponse.addEventListener("change", () => { responseFields.hidden = !includeResponse.checked; });
    form.append(field("Decision", decision), field("Action ID", action), field("Arguments as a JSON object", args));
    const responseToggle = element("label", null, "check");
    responseToggle.append(includeResponse, document.createTextNode("Annotate a response contract"));
    form.append(responseToggle, responseFields, field("Presentation intent", presentation));
    caseBox.append(form);
    current.form = { decision, action, args, includeResponse, responseReq, responseLang, presentation };
  }
  function renderLanguageForm(item) {
    const form = element("div", null, "panel");
    const qualificationItems = item.language === "ja"
      ? [["native_japanese","Native Japanese"],["fluent_japanese","Fluent Japanese"]]
      : [["fluent_bilingual","Fluent English/Japanese bilingual"]];
    const qualification = select(qualificationItems);
    const naturalness = select([["approved","Approved"],["needs_revision","Needs revision"]]);
    const meaning = select([["approved","Approved"],["needs_revision","Needs revision"]]);
    form.append(field("Language qualification", qualification));
    form.append(field("Naturalness in this language", naturalness));
    form.append(field("Preserves intended meaning", meaning));
    caseBox.append(form);
    current.form = { qualification, naturalness, meaning };
  }
  function render() {
    errorBox.textContent = "";
    if (session.completed >= session.total) {
      caseBox.replaceChildren(element("h2", "Review session complete"));
      caseBox.append(element("p", "All assigned cases have a saved review record. The case inventory remains a draft until its provenance and readiness gates pass."));
      return;
    }
    const available = remaining();
    if (!available.length) {
      skipped.clear();
      return render();
    }
    current = available[0];
    renderInputs(current);
    const toolbar = element("div", null, "toolbar");
    const skip = element("button", "Skip this case", "secondary"); skip.type = "button";
    const submit = element("button", session.role === "semantic" ? "Save independent annotation" : "Save language review", "primary"); submit.type = "button";
    skip.addEventListener("click", () => { skipped.add(current.index); render(); });
    submit.addEventListener("click", submitCurrent);
    toolbar.append(skip, submit);
    caseBox.append(toolbar);
  }
  async function submitCurrent() {
    errorBox.textContent = "";
    try {
      let payload;
      if (session.role === "semantic") {
        const f = current.form, decision = f.decision.value;
        let argumentsValue = {};
        try { argumentsValue = JSON.parse(f.args.value); }
        catch { throw new Error("Arguments must be valid JSON."); }
        if (!argumentsValue || Array.isArray(argumentsValue) || typeof argumentsValue !== "object")
          throw new Error("Arguments must be a JSON object.");
        const proposed = {
          decision,
          action: decision === "act" ? f.action.value.trim() : null,
          arguments: decision === "act" ? argumentsValue : {},
          requires_clarification: decision === "clarify",
        };
        if (decision === "act" && !proposed.action) throw new Error("Enter the capability ID for an act decision.");
        if (f.includeResponse.checked) {
          proposed.response_contract = { requirement: f.responseReq.value, language: f.responseLang.value };
        }
        if (f.presentation.value) proposed.presentation_intent = f.presentation.value;
        payload = { index: current.index, proposed_gold: proposed };
      } else {
        const f = current.form;
        payload = {
          index: current.index,
          language_review: {
            qualification: f.qualification.value,
            naturalness_status: f.naturalness.value,
            meaning_preservation_status: f.meaning.value,
          },
        };
      }
      await getJson("/api/review", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Lumi-Review-Token": session.csrf_token,
        },
        body: JSON.stringify(payload),
      });
      session = await getJson("/api/session");
      completed = new Set(session.completed_indices);
      skipped.clear();
      render();
    } catch (error) {
      errorBox.textContent = error.message;
    }
  }
  async function start() {
    try {
      session = await getJson("/api/session");
      cases = await getJson("/api/cases");
      completed = new Set(session.completed_indices);
      sessionBox.textContent = session.role + " review · pseudonymous reviewer " + session.reviewer_id + " · " + session.completed + "/" + session.total + " saved";
      render();
    } catch (error) {
      sessionBox.textContent = "Could not load the local review session.";
      errorBox.textContent = error.message;
    }
  }
  start();
})();
</script>
</body>
</html>
"""


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], manager: ReviewManager) -> None:
        super().__init__(address, ReviewRequestHandler)
        self.manager = manager
        self.csrf_token = secrets.token_urlsafe(32)
        self.nonce = secrets.token_urlsafe(18)


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: Any) -> None:
        # Never write case or reviewer data to the console access log.
        return

    def _base_url(self) -> str:
        port = self.server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; "
            f"script-src 'nonce-{self.server.nonce}'; "
            f"style-src 'nonce-{self.server.nonce}'; "
            "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _check_host(self) -> bool:
        return self.headers.get("Host") == f"127.0.0.1:{self.server.server_address[1]}"

    def do_GET(self) -> None:
        if not self._check_host():
            return self._json(403, {"error": "Use the exact loopback URL printed by the workbench."})
        path = urlsplit(self.path).path
        if path == "/":
            body = PAGE.replace("__NONCE__", self.server.nonce).encode("utf-8")
            return self._send(200, body, "text/html; charset=utf-8")
        if path == "/api/session":
            manager = self.server.manager
            completed = manager.completed_indices()
            return self._json(200, {
                "role": manager.role,
                "reviewer_id": manager.reviewer_id,
                "total": len(manager.assigned),
                "completed": len(completed),
                "completed_indices": completed,
                "csrf_token": self.server.csrf_token,
            })
        if path == "/api/cases":
            return self._json(200, self.server.manager.safe_cases())
        return self._json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self._check_host():
            return self._json(403, {"error": "Use the exact loopback URL printed by the workbench."})
        if self.headers.get("Origin") != self._base_url():
            return self._json(403, {"error": "Cross-origin submissions are not accepted."})
        if self.headers.get("X-Lumi-Review-Token") != self.server.csrf_token:
            return self._json(403, {"error": "The local review token is invalid."})
        if urlsplit(self.path).path != "/api/review":
            return self._json(404, {"error": "Not found."})
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "Invalid content length."})
        if size <= 0 or size > 65536:
            return self._json(413, {"error": "Review submission must be between 1 byte and 64 KiB."})
        if self.headers.get_content_type().lower() != "application/json":
            return self._json(415, {"error": "Review submission must use application/json."})
        try:
            payload = _strict_json_object(self.rfile.read(size))
            result = self.server.manager.submit(payload)
        except (WorkbenchError, ReviewRecordError) as exc:
            return self._json(400, {"error": str(exc)})
        except OSError:
            return self._json(500, {"error": "Could not safely append the review record."})
        self._json(200, result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect blind Lumi development-case annotations on loopback only.")
    parser.add_argument("--cases", type=Path, required=True, help="Controlled development cases JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Controlled review ledger JSONL outside the Git repository.")
    parser.add_argument("--reviewer-id", required=True, help="Stable pseudonym in the form rev-example-01; do not use a name.")
    parser.add_argument("--role", choices=("semantic", "language"), required=True)
    parser.add_argument("--port", type=int, default=0, help="Loopback TCP port; 0 selects an ephemeral port.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not REVIEWER_ID_PATTERN.fullmatch(args.reviewer_id):
        raise WorkbenchError("reviewer ID must be a stable pseudonym like rev-reviewer-01")
    if not 0 <= args.port <= 65535:
        raise WorkbenchError("port must be between 0 and 65535")
    cases_path = args.cases.expanduser().resolve()
    output_path = _safe_output_path(args.output, cases_path)
    cases = _read_cases(cases_path)
    manager = ReviewManager(cases, output_path, args.reviewer_id, args.role)
    with ReviewHTTPServer(("127.0.0.1", args.port), manager) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(
            f"Loaded {len(manager.assigned)} development cases for {args.role} review. "
            f"Ledger: {output_path}\nOpen {url} (Ctrl+C stops the local server)."
        )
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLocal review server stopped.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkbenchError, ReviewRecordError, EvaluationInputError) as exc:
        print(f"review workbench: {exc}", file=sys.stderr)
        raise SystemExit(2)
