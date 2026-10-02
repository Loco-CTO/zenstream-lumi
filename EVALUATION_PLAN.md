# Lumi behavioral evaluation plan

**Status:** Evaluation framework and dependency-free scorer implemented; no ready benchmark cases, scores, or trained candidates yet. See [evaluation/README.md](evaluation/README.md).

**Before major compute:** A versioned development set and separate sealed final holdout must be authored, reviewed, and frozen.

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

Examples must be semantically varied, not mostly paraphrases. Split by intent pattern, source, title/entity, conversation family, and generation template where applicable so near-duplicates cannot cross into the final holdout. Keep every benchmark's source, license, revision, and use in the provenance manifest.

## Initial engineering thresholds

These targets come from the user goal. They are minimum targets, not claims about Lumi performance.

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

Track false state-changing actions separately from missed actions. A false state-changing action means the model proposes an action when the gold label calls for no action, discussion, or clarification. Report raw counts, denominators, per-category scores, and confidence intervals; do not let aggregate accuracy hide a failing safety-critical slice. For a 99.9% validity target, a small sample cannot establish the target precisely: use a sufficiently large, frozen test and report a one-sided confidence bound as well as observed rate. As a rough minimum statistical resolution, zero invalid responses in about 2,995 independent trials gives a one-sided 95% binomial upper bound near a 0.1% error rate; that does not replace broad semantic coverage or per-slice checks.

## Scoring and runtime records

The machine-readable case and prediction formats are in `evaluation/`. The scorer uses a canonical semantic adapter; it does not select Lumi's runtime wire protocol. Raw output is retained so malformed responses count against structural validity. Structural validity, semantic exact match, argument extraction, and false state-changing actions are separate metrics. Unsupported factual claims require a separately recorded human grounding review.

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
