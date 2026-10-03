# Lumi behavioral evaluation plan

**Status:** Evaluation case/output/prediction formats, a draft capability registry, a model-independent candidate-input projection, a provenance-gated callback runner with bounded fixture-only tool trajectories, a required-slice coverage audit, and a dependency-free scorer implemented. A 101-case development draft is held in local controlled storage and covers all required slice-language pairs, but has zero ready cases, no independent reviews, no approved provenance, and no final holdout. See [evaluation/README.md](evaluation/README.md) and the metadata-only [draft inventory manifest](evaluation/draft_inventory_manifest.json).

Case authors and reviewers must follow the draft [annotation guide](evaluation/ANNOTATION_GUIDE.md). Case schema v4 can carry static trusted-context fixtures, an ordered fixture-only tool scenario, and reviewed response, presentation, and expected tool-trajectory labels. The candidate receives declared tool schemas and results only after matching calls; the simulator invokes no real services and applies no state changes. Candidate messages receive separate human review for language, naturalness, meaning, relevance, and concision; they are not compared to one reference string. The capability registry maps a first set of candidate outcomes to a pinned Orchestrator contract and route implementation, but remains unapproved. Its current gaps include explicit catalog filters for genre/year/runtime/watched state and the missing Lumi playback gateway. Do not label cases for these unreviewed or unimplemented outcomes as ready.

**Before major compute:** A versioned development set and separate sealed final holdout must be authored, reviewed, and frozen.

Run `python evaluation/coverage_audit.py --cases <cases.jsonl>` while cases are being authored. It reports ready, draft, and excluded case presence by required slice, language, and family without scoring a model. A slice-language pair counts as present when it has at least one ready case; this is an inventory signal only and does not establish adequate sample size or linguistic quality. Auditing the sealed final holdout requires both `--split final_holdout` and `--final-audit`.

## Required slices

Report the overall score and every slice below, in English, natural Japanese, and English/Japanese code-switching where applicable. Avoid relying on translated English as the Japanese test set.

- Simple positive action selection and exact argument extraction.
- Search intent, semantic discovery, and constraints such as watched state, genre, year, and runtime.
- No-action cases: negation, questions about capability, hypothetical or future intent, discussion, troubleshooting, and quoted commands.
- Ambiguous references and clarification decisions.
- Multi-turn follow-ups, omitted subjects, corrections, interruptions, and conflicting constraints.
- English slang, typos, incomplete grammar, shorthand, and voice-command phrasing.
- Natural Japanese: informal and abbreviated speech, omitted subjects, kana/kanji variants, casual commands, titles in Japanese, and English titles embedded in Japanese.
- Code-switching in both directions, including English phrases inside Japanese and Japanese phrases inside English.
- Grounding/tool cases: available and unavailable media, authoritative tool results, failures, and unavailable or conflicting state.
- Structured response validity, unsupported factual claims, and disallowed output content.

For English/Japanese cases, score English-matrix and Japanese-matrix code-switching separately as well as together. Human review must evaluate linguistic naturalness and preservation of the intended meaning as distinct checks. The controlled review ledger stores independent proposed labels, response contracts, presentation intent, expected tool paths, language assessments, and any adjudication, each bound to the case's canonical content hash. No human review records currently exist. Case v4 hashes inline trusted-context fixtures, tool definitions, ordered tool outcomes, and the expected trajectory with the rest of each sample. The callback runner selects ready cases only after ledger and sample-provenance checks. It supplies only turns, trusted context, declared tool schemas, and prior observations to candidate code; final-holdout execution is blocked unless explicitly enabled. It invokes no real tools and applies no state changes. Candidate messages are scored separately for response presence, reviewed language and quality, and presentation intent; fixture-based tool paths have separate completion, exact-path, precision/recall, unknown-tool, and fixture-miss metrics. No candidate messages or human response reviews currently exist, and interactive tool fixtures do not establish end-to-end runtime behavior.

Examples must be semantically varied, not mostly paraphrases. Split by intent pattern, source, title/entity, conversation family, and generation template where applicable so near-duplicates cannot cross into the final holdout. Keep every benchmark's source, license, revision, and use in the provenance manifest.

## Initial engineering thresholds

These targets come from the user goal. They are minimum targets, not claims about Lumi performance.

See the [sample-size plan](evaluation/SAMPLE_SIZE_PLAN.md) for the minimum independent-case counts needed even when every scored output is correct, and the limits of interpreting those counts.

| Metric | Initial target |
|---|---:|
| Structured-response validity | >= 99.9% |
| Simple action selection | >= 97% |
| Argument extraction | >= 95% |
| Negation/no-action correctness | >= 98% |
| English ZenStream tasks | >= 97% |
| Japanese ZenStream tasks | >= 95% |
| English/Japanese code-switching | >= 94% |
| Multi-turn reference resolution | >= 92% |
| Difficult ambiguous requests | >= 90% |

Track false actions and missed actions separately. With v1, a false action is any predicted `act` when the gold label is `no_action`, `respond`, or `clarify`; this includes unnecessary read-only calls. The scorer now reports false state-changing action and unclassified action rates from the capability registry, but only after every capability and the registry itself are approved. While the checked-in registry remains draft, both metrics must say `not_measured`; never report a zero rate. The state-changing metric counts an unmatched registered mutation across all selected cases. The unclassified metric counts predictions outside the approved action vocabulary among valid `act` predictions. Report raw counts, denominators, per-category scores, and confidence intervals; do not let aggregate accuracy hide a failing safety-critical slice. For a 99.9% validity target, a small sample cannot establish the target precisely: use a sufficiently large, frozen test and report a one-sided confidence bound as well as observed rate. As a rough minimum statistical resolution, zero invalid responses in about 2,995 independent trials gives a one-sided 95% binomial upper bound near a 0.1% error rate; that does not replace broad semantic coverage or per-slice checks.

## Scoring and runtime records

The machine-readable case, output, trajectory-step, and prediction formats are in `evaluation/`. The scorer uses a canonical evaluation adapter; it does not select Lumi's runtime wire protocol. Raw output is retained so malformed responses count against structural validity. Structural validity, semantic exact match, argument extraction, per-slot precision/recall, clarification precision/recall, response-contract compliance, human-reviewed response quality, presentation intent, fixture-only tool trajectories, false-action rate, and registry-classified mutation errors are separate metrics. Tool trajectory metrics include exact ordered paths, final completion, call precision/recall, unknown-tool calls, and fixture misses. Response language, naturalness, meaning preservation, intent appropriateness, and concision are reported only for manually reviewed candidate messages, with review coverage shown separately. Argument slots are the top-level argument properties scoped by action ID; missing and extra properties reduce recall and precision respectively, while a wrong value counts against both. The v1 false-action rate counts every predicted `act` on a `no_action`, `respond`, or `clarify` gold case, including unnecessary read-only calls. False state-changing actions and unclassified action IDs are scored only against an approved, hashed capability registry; with the current draft registry, those measures remain `not_measured`. Unsupported factual claims require a separately recorded human grounding review.

For each candidate, report:

- Exact intent/action match; false-action rate; clarification precision/recall.
- Exact slot match and per-slot precision/recall for entities, constraints, and references.
- Unsupported factual claims per response and severity; distinguish unsupported statements from grounded tool results.
- Schema-valid response rate and semantic correctness as separate metrics.
- Overall and English/Japanese/code-switch scores, including per-category counts and confidence intervals.
- Parameters, tokenizer/vocabulary size, model artifact bytes, quantization, and training token count.
- Cold start, model-load time, warm latency, time to first useful output, and total latency for short representative requests.
- CPU and GPU throughput where available; peak/idle RAM and VRAM; idle CPU/GPU activity; unload and wake/reload times.

Benchmark on named and reproducible hardware/software configurations. Compare CPU-only against acceleration; never infer GPU benefit from peak throughput alone.

## Integrity and safety controls

- Keep development data available for iteration and final holdout data sealed from candidate training, synthetic-data prompts, filtering, and hyperparameter selection.
- Store data hashes, versioned annotation guidelines, prompt families, model/runtime versions, and scorer versions for each run.
- Do not hard-code cases, weaken scoring, remove hard categories, or lower targets to pass.
- Make no state-changing action authoritative in Lumi. The ZenStream server must validate identity, permissions, media existence, constraints, and current state before acting.
- Require native-speaker review of Japanese naturalness and intended meaning. Track disagreements and adjudication rather than silently translating or normalizing them away.
