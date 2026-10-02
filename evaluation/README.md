# Lumi evaluation harness

**Status:** Versioned case/prediction formats, a draft capability registry, a dependency-free scorer, a model-independent input projection, a provenance-gated callback runner, and a required-slice coverage audit exist. There are no ready benchmark cases, model backend, batch candidate CLI, predictions, or Lumi scores yet.

The canonical semantic record is an evaluation adapter, not a decision about Lumi's eventual runtime or ZenStream integration protocol. It gives competing model formulations one comparable representation for action/no-action choice, action name, arguments, and clarification. A later protocol adapter may map between this record and the deployed interface.

## Files and use

The draft capability vocabulary lives in capabilities.json. It records read-only versus state-changing effects, integration maturity, argument schemas, operation mappings, known limitations, and the exact Orchestrator contract and implementation snapshot. It is an annotation resource, not an executable tool allow-list or Lumi runtime protocol. Validate it with python evaluation/capability_registry.py. Action-bearing benchmark cases remain drafts until their capability definitions are reviewed.

- `case.schema.json` defines version 2 authored benchmark cases. A case carries an immutable family ID, language and category labels, conversation turns, a required nullable `trusted_context` fixture, a gold semantic record, and a provenance-record ID. The fixture has its own version and structured status/payload records; it is static test input, not tool execution or a runtime protocol.
- `input_adapter.py` projects a validated case into a versioned candidate input containing only `turns` and `trusted_context`. It leaves out the gold answer, language/category labels, IDs, split, review, and provenance metadata, and deep-copies the candidate-visible values. It does not invoke a model or execute tools.
- `candidate_runner.py` invokes an in-process `Candidate.predict(...)` callback on ready cases only after schema/readiness and provenance checks, then wraps the untouched raw output for scoring. Final-holdout execution requires explicit opt-in. It is synchronous and does not load a model or run tools.
- `prediction.schema.json` wraps the raw model output. Keeping the original string means malformed JSON and schema-invalid output count as failures instead of disappearing during preprocessing.
- `thresholds.json` contains the initial targets from the goal. It is not editable to make a candidate pass.
- `required_slices.json` defines the stable case-category IDs and language applicability for the required evaluation-plan slices.
- `coverage_audit.py` validates a case inventory and reports ready, draft, and excluded presence for every required slice/language pair, including distinct family counts. It does not score model output or assess sample sufficiency or quality. It defaults to development and requires explicit final-audit opt-in for holdout inventory.
- `scorer.py` is a Python standard-library-only scorer. It reports structural validity separately from semantic exact match, language/category slices, argument extraction, clarification decisions, false actions, and manually reviewed unsupported factual claims. It reports clarification precision/recall and micro/per-action-slot argument precision/recall. A slot is one top-level argument property scoped by action ID; a wrong value counts as both a false positive and a missed gold slot. The v1 false-action rate counts any predicted `act` on a non-`act` gold case, including read-only calls. False state-changing action and unclassified action rates use the capability registry only after the entire registry is approved; while it remains draft, both rates are explicitly `not_measured`, never reported as zero. The state-changing rate counts unmatched registered mutations over all selected cases, and the unclassified rate counts action IDs outside the approved registry among valid predicted action decisions. Scoring requires sample-level provenance records and validates their source/generation joins and use permissions first.
- The direct `score_records(...)` API takes the source and synthetic-generation manifests as required inputs, just like file-based scoring; callers cannot bypass source-level use-permission checks by supplying sample records alone.
- `tests/test_evaluation_scorer.py` checks only scorer behavior. Its inline cases are software fixtures, not Lumi benchmark or training data.
- `tests/test_evaluation_input_adapter.py` checks the candidate-input projection and mutation isolation; its inline cases are software fixtures, not Lumi benchmark or training data.
- `tests/test_evaluation_candidate_runner.py` checks provenance/readiness gates, final-holdout opt-in, and raw-output preservation; its inline cases are software fixtures, not Lumi benchmark or training data.
- `tests/test_evaluation_coverage_audit.py` checks inventory validation, split isolation, review-state counts, and slice/language presence; its inline cases are software fixtures, not Lumi benchmark or training data.

To score a reviewed development set:

```powershell
python evaluation/scorer.py --cases <development-cases.jsonl> --predictions <candidate-predictions.jsonl> --provenance-records <document-records.jsonl> --capabilities evaluation/capabilities.json --split development --candidate-version <id> --output <report.json>
```

To inspect development-set coverage before scoring:

```powershell
python evaluation/coverage_audit.py --cases <development-cases.jsonl> --output <coverage-report.json>
```

The audit uses the fixed category IDs in `required_slices.json`; any extra descriptive category tags are listed separately. A `ready_cases_present` result means at least one approved case exists in that slice and language. It is not a model score, a minimum-size check, or a substitute for independent native-speaker and semantic review. Pass a single-split inventory; the coverage audit rejects rows assigned to the other split. The scorer checks family leakage when it validates the full evaluation inventory. For a sealed holdout inventory, pass `--split final_holdout --final-audit` only in the controlled final-audit workflow. The flag is a workflow checkpoint, not filesystem access control; keep holdout files in controlled storage.

Each prediction row must include the raw output as a string. Grounding review is a separate optional annotation; when present, it records the number of factual claims and the unsupported subset. The scorer reports its review coverage so a low unsupported-claim rate cannot hide unaudited answers.

Scoring `final_holdout` requires both `--split final_holdout` and `--final-audit`. That flag is an intentional checkpoint, not an access-control boundary: holdout files should remain in controlled storage and should only be scored after the candidate is frozen. Do not use final-holdout results for training, data generation, filtering, architecture selection, or repeated tuning.

Cases are scoreable only when `review_status` is `ready`, their annotation is approved, and Japanese or English/Japanese cases have native-language review marked approved. The scorer refuses a selected split containing drafts. It also refuses duplicate IDs, predictions for other splits, case families split across development and holdout, and provenance records that are missing, reused, unapproved, assigned to another split, or hash a different case payload. File-based reports include hashes for the case file, predictions, sample records, source/generation manifests, and capability registry. String argument comparison uses Unicode NFC, whitespace collapse, and case folding; list order and punctuation remain significant.

Structural validity is evaluated from the raw response. Semantic exact match, action selection, and slot extraction are separate. The report includes two-sided 95% Wilson intervals. For the 99.9% structural-validity target it also reports a one-sided exact binomial upper bound on the error rate; zero invalid outputs in 2,995 independent trials is the approximate minimum for that bound to reach 0.1%.

The callback runner has no model backend or batch CLI and does not simulate stateful tool trajectories. The capability registry is draft: catalog search has no explicit genre/year/runtime/watched filters, the generated OpenAPI enum omits the recommendations value accepted by the runtime Home route, and no Lumi playback gateway exists. The current directory still has no benchmark cases, candidate predictions, or behavioral evidence. Do not interpret software fixtures or example snippets as model results. The eventual development set and sealed final holdout still need semantically diverse English, natural Japanese, code-switching, multi-turn, grounding, and no-action coverage plus full provenance records.
