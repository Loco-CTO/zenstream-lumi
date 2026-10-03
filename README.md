# Lumi

Lumi is a proposed optional, local natural-language capability for ZenStream. It is developed and released independently in this repository. ZenStream must remain fully useful when Lumi is absent, disabled, or unavailable.

## Project status

This repository contains Lumi's research, evaluation, and provenance foundation plus two level-1 exploratory random-init experiments: a structured-prediction smoke and a byte-level conversational-response smoke. The second experiment proves that a tiny local model can train, save, reload, and generate bounded UTF-8 text on CPU; its two illustrative development generations show clear response-selection failures and have no human quality review. Neither experiment is a product model. No release candidate, production inference service, approved production-quality training corpus, or selected runtime exists. Temporary examples, model artifacts, and logs remain in local controlled storage and are not checked in. No pretrained model weights are permitted for Lumi; any eventual model must start from random initialization.

Architecture, tokenizer, model size, runtime, training corpus, integration protocol, and artifact licensing remain open research decisions. Research claims are not Lumi measurements. See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the initial evidence review and its limitations.

## Product boundaries

- Lumi is opt-in. A disabled installation must not download a model, start an inference process, reserve meaningful RAM or VRAM, or create AI-specific CPU/GPU work.
- Lumi interprets language and can suggest structured intent. ZenStream remains authoritative for media, permissions, playback, metadata, user state, and all state-changing validation.
- Lumi must also provide concise, natural English, Japanese, and appropriate English/Japanese code-switch responses for greetings, acknowledgements, harmless conversation, media comments, capability questions, ambiguity, uncertainty, and unsupported requests. An intent classifier by itself is not a complete Lumi system.
- Lumi must not emit executable application code, SQL, shell commands, or arbitrary system commands as part of normal interaction.
- CPU inference is required. GPU use is optional and must show a measurable benefit on supported hardware.
- Initial language targets are English, Japanese, and English/Japanese code-switching. Japanese evaluation must be natural Japanese and reviewed by competent speakers.
- Data provenance, licensing, benchmark integrity, and reproducibility are release requirements.

## Evidence levels

- **Level 1 — exploratory:** small local experiments may use temporary material with a clearly recorded permission scope and a separate exploratory development split. Keep data and model artifacts local. These runs answer engineering questions and cannot support product-quality claims.
- **Level 2 — development:** compare reproducible candidates using permitted, provenance-recorded training material and a frozen, independently reviewed development set, including qualified Japanese review. The final holdout and final statistical power are not prerequisites for this stage.
- **Level 3 — release/final:** require admitted training sources, qualified reviewed evaluation, a sequestered and statistically suitable holdout, resolved artifact-distribution terms, production hardware measurements, and complete release provenance.

Level-3 requirements must not block clearly scoped level-1 work. The [structured intent smoke](experiments/RANDOM_INIT_INTENT_SMOKE.md) and [conversation response smoke](experiments/CONVERSATION_RESPONSE_SMOKE.md) are engineering evidence only; their examples are not admitted as production training data or formal benchmark cases.

## Project records

- [Evaluation plan](EVALUATION_PLAN.md): target thresholds, error categories, measurement, and holdout rules.
- [Evaluation sample-size plan](evaluation/SAMPLE_SIZE_PLAN.md): statistical floors for estimating the current target thresholds, with the assumptions and annotation cost made explicit.
- [Evaluation tools](evaluation/README.md): canonical case/output/prediction formats, a hash-bound independent-review ledger, a blind loopback review workbench and draft-only ledger importer, a candidate-input projection, provenance/review-gated runner, coverage audit, and standard-library scorer; the controlled draft inventory has no human review records or ready cases yet.
- [Annotation guide](evaluation/ANNOTATION_GUIDE.md): draft labeling, response contracts and presentation intent, the controlled review ledger, Japanese/code-switch review, split grouping, and v4 coverage limits.
- [Research notes](RESEARCH_NOTES.md): current findings, limitations, open hypotheses, and proposed experiments.
- [Research references](RESEARCH_REFERENCES.md): sources that influence design; these are not training data.
- [Tokenizer pilot protocol](experiments/TOKENIZER_PILOT.md): a controlled EN/JA/code-switch tokenizer comparison, gated on approved text and reviewed development cases.
- [Model formulation and architecture pilot](experiments/ARCHITECTURE_PILOT.md): a staged, compute-matched comparison of conditional-generation formulations and short-context sequence architectures, gated on rights, review, tokenizer, and hardware evidence.
- [Random-init intent smoke](experiments/RANDOM_INIT_INTENT_SMOKE.md): first executable CPU train/save/reload/infer evidence, with its tiny local data scope, measured failures, and limits.
- [Exploratory experiment dependencies](experiments/requirements-exploratory.txt): pinned NumPy dependency for local prototype experiments only.
- [Data sources](DATA_SOURCES.md): inclusion policy and human-readable provenance summary.
- [MASSIVE 1.1 source audit](experiments/MASSIVE_AUDIT.md): pinned archive, file hashes, count reconciliation, licensing evidence, quality signals, and limits; the source remains unadmitted.
- [Common Pile v0.1 source audit](experiments/COMMON_PILE_AUDIT.md): release-level review of the 30 reported source groups, rights caveats, English-focused filtering, and lack of demonstrated Japanese/code-switch coverage; no content is admitted.
- [JMRD and RecomMind Japanese dialogue source audit](experiments/JMRD_RECOMMIND_AUDIT.md): comparison of two Japanese human-human movie recommendation releases, their task fit, repository/paper count discrepancy, upstream metadata and review lineage, CC-BY-SA declarations, and unresolved participant/privacy gates; neither source is admitted.
- [Natural Japanese-English code-switch source audit](experiments/NATURAL_CODE_SWITCH_AUDIT.md): distinguishes parallel translations from real mixed-language messages, records BSD's CC BY-NC-SA terms, the privacy limits of a small iMessage study, and a 2025 pseudo-code-switch method-only lead; no reviewed source is admitted.
- [CodeMixBench coverage and provenance audit](experiments/CODEMIXBENCH_AUDIT.md): checks the 2025 benchmark's language-pair table and synthetic-data method; none of its listed pairs is Japanese-English and no benchmark data is admitted.
- [Capability and playback boundary audit](experiments/CAPABILITY_BOUNDARY_AUDIT.md): pinned Orchestrator and client source evidence for grant-filtered resolution, playback negotiation, and client-owned execution; the registry remains draft.
- [Human contributor intake checklist](provenance/AUTHOR_CONTRIBUTION_CHECKLIST.md): separates evaluation-only contributions from training grants and records exact model-distribution scope before collection.
- [Machine-readable provenance](provenance/data_sources.json): records the single approved ASDC diagnostic item; MASSIVE remains a candidate without admitted content. [The validator](provenance/README.md#provenance-gate) checks source/sample joins and split-safe evaluation hashes.
- [Synthetic-data provenance](provenance/synthetic_data.json): currently empty because no synthetic data has been generated or used.
- [Model card](MODEL_CARD.md): pre-release status and limitations; no complete Lumi model has been selected.
- [Third-party notices](THIRD_PARTY_NOTICES.md): current code license and outstanding artifact-license research.
- [Experiment record template](experiments/EXPERIMENT_TEMPLATE.md): reproducibility and comparison fields.
- [Conversation response smoke](experiments/CONVERSATION_RESPONSE_SMOKE.md): a random-init byte-level sequence-generation pilot, its two held-out illustrative outputs, and the reasons it is not a quality-bearing candidate.

## Initial sequence

1. Grow the executable level-1 path through small experiments that diagnose observed failures, using only clearly scoped local data and keeping artifacts outside Git.
2. Build and review a semantically diverse English/Japanese/code-switch development set; qualify Japanese review before using results for development decisions. Build and seal the final holdout for level-3 claims.
3. Audit candidate public and generated data for provenance, permitted use, and contamination risk before admitting it to development-quality or release training.
4. Compare task-focused formulations, tokenizers, sequence architectures, and training methods with controlled, compute-matched level-2 experiments once the reviewed development set is ready; do not wait for the final holdout to run small exploratory tests.
5. Benchmark model quality and user-visible latency, cold start, memory, unload, and wake-up behavior on each target hardware class.
6. Select the smallest candidate that passes the behavioral targets and has the best measured product experience; retain evidence and rejected alternatives.

No production model or ZenStream integration is implied by this research scaffold.
