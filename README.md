# Lumi

Lumi is a proposed optional, local natural-language capability for ZenStream. It is developed and released independently in this repository. ZenStream must remain fully useful when Lumi is absent, disabled, or unavailable.

## Project status

This repository contains Lumi's research, evaluation, and provenance foundation plus several level-1 random-init diagnostics: a structured-intent smoke; a conversational-response smoke and four-seed follow-up; Japanese travel-dialog state/response runs at 64/24 and 512/64 dialogues; a small paired English/Japanese MASSIVE intent/slot pilot; a text-only JECS Japanese/English/code-switch byte-language-model ablation; and a JECS tokenizer intrinsic comparison. The tokenizer candidates are matched for intrinsic measurements, but the separate model pilots use different tasks and data scopes; none establishes product quality. The structured and dialogue pilots found substantial generalization failures; the JECS runs measure tokenization and byte prediction on acted/read transcripts, not natural responses. Reports record their exact limits. Level 2 has not started because no independently reviewed development set is ready. No release candidate, production inference service, approved production-quality training corpus, or selected runtime exists. Temporary examples, model artifacts, tokenizers, and logs remain in local controlled storage and are not checked in. No pretrained model weights are permitted for Lumi; any eventual model must start from random initialization.

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

Level-3 requirements must not block clearly scoped level-1 work. The intent, response, JMultiWOZ, MASSIVE, and JECS reports are narrow engineering or source diagnostics; none is the formal independently reviewed Lumi evaluation set or evidence that the product targets pass.

## Project records

- [Evaluation plan](EVALUATION_PLAN.md): target thresholds, error categories, measurement, and holdout rules.
- [Evaluation sample-size plan](evaluation/SAMPLE_SIZE_PLAN.md): statistical floors for estimating the current target thresholds, with the assumptions and annotation cost made explicit.
- [Evaluation tools](evaluation/README.md): canonical case/output/prediction formats, a hash-bound independent-review ledger, a blind loopback review workbench and draft-only ledger importer, a candidate-input projection, provenance/review-gated runner, coverage audit, and standard-library scorer; the controlled draft inventory has no human review records or ready cases yet.
- [Annotation guide](evaluation/ANNOTATION_GUIDE.md): draft labeling, response contracts and presentation intent, the controlled review ledger, Japanese/code-switch review, split grouping, and v4 coverage limits.
- [Research notes](RESEARCH_NOTES.md): current findings, limitations, open hypotheses, and proposed experiments.
- [Research references](RESEARCH_REFERENCES.md): sources that influence design; these are not training data.
- [Tokenizer pilot protocol](experiments/TOKENIZER_PILOT.md): a controlled EN/JA/code-switch tokenizer comparison, gated on approved text and reviewed development cases.
- [JECS tokenizer intrinsic pilot](experiments/JECS_TOKENIZER_INTRINSIC_PILOT.md): a local, source-scoped byte/BPE/Unigram measurement that selects no tokenizer.
- [Model formulation and architecture pilot](experiments/ARCHITECTURE_PILOT.md): a staged, compute-matched comparison of conditional-generation formulations and short-context sequence architectures, gated on rights, review, tokenizer, and hardware evidence.
- [Random-init intent smoke](experiments/RANDOM_INIT_INTENT_SMOKE.md): first executable CPU train/save/reload/infer evidence, with its tiny local data scope, measured failures, and limits.
- [Exploratory experiment dependencies](experiments/requirements-exploratory.txt): pinned NumPy dependency for local prototype experiments only.
- [Tokenizer pilot dependency](experiments/requirements-tokenizer-pilot.txt): pinned SentencePiece dependency for the scoped local intrinsic pilot.
- [Data sources](DATA_SOURCES.md): inclusion policy and human-readable provenance summary.
- [MASSIVE 1.1 source audit](experiments/MASSIVE_AUDIT.md): pinned archive, file hashes, count reconciliation, licensing evidence, quality signals, and limits; its only admitted use is the separately scoped local diagnostic described in [the intent/slot pilot](experiments/MASSIVE_INTENT_SLOT_PILOT.md).
- [Common Pile v0.1 source audit](experiments/COMMON_PILE_AUDIT.md): release-level review of the 30 reported source groups, rights caveats, English-focused filtering, and lack of demonstrated Japanese/code-switch coverage; no content is admitted.
- [JMRD and RecomMind Japanese dialogue source audit](experiments/JMRD_RECOMMIND_AUDIT.md): comparison of two Japanese human-human movie recommendation releases, their task fit, repository/paper count discrepancy, upstream metadata and review lineage, CC-BY-SA declarations, and unresolved participant/privacy gates; neither source is admitted.
- [Natural Japanese-English code-switch source audit](experiments/NATURAL_CODE_SWITCH_AUDIT.md): distinguishes parallel translations from real mixed-language messages, records BSD's CC BY-NC-SA terms, the privacy limits of a small iMessage study, and a 2025 pseudo-code-switch method-only lead; no reviewed source is admitted.
- [CodeMixQA Japanese-English source audit](experiments/CODEMIXQA_AUDIT.md): records the synthetic QA task, the paper/card data-license mismatch, provenance gaps, and why it remains a research reference without admitted data.
- [CodeMixBench coverage and provenance audit](experiments/CODEMIXBENCH_AUDIT.md): checks the 2025 benchmark's language-pair table and synthetic-data method; none of its listed pairs is Japanese-English and no benchmark data is admitted.
- [Capability and playback boundary audit](experiments/CAPABILITY_BOUNDARY_AUDIT.md): pinned Orchestrator and client source evidence for grant-filtered resolution, playback negotiation, and client-owned execution; the registry remains draft.
- [Human contributor intake checklist](provenance/AUTHOR_CONTRIBUTION_CHECKLIST.md): separates evaluation-only contributions from training grants and records exact model-distribution scope before collection.
- [Machine-readable provenance](provenance/data_sources.json): records the exact ASDC evaluation item and the narrowly scoped JMultiWOZ, MASSIVE, and JECS local pilots; none is a broad pretraining or tokenizer corpus. [The validator](provenance/README.md#provenance-gate) checks source/sample joins and split-safe evaluation hashes.
- [Synthetic-data provenance](provenance/synthetic_data.json): currently empty because no synthetic data has been generated or used.
- [Model card](MODEL_CARD.md): pre-release status and limitations; no complete Lumi model has been selected.
- [Third-party notices](THIRD_PARTY_NOTICES.md): current code license and outstanding artifact-license research.
- [Experiment record template](experiments/EXPERIMENT_TEMPLATE.md): reproducibility and comparison fields.
- [Conversation response smoke](experiments/CONVERSATION_RESPONSE_SMOKE.md): a random-init byte-level sequence-generation pilot, its two held-out illustrative outputs, and the reasons it is not a quality-bearing candidate.
- [JMultiWOZ state/response pilots](experiments/JMULTIWOZ_PILOT.md) and [scale follow-up](experiments/JMULTIWOZ_SCALE_PILOT.md): two local Japanese travel-dialog diagnostics, neither a ZenStream task benchmark.
- [MASSIVE intent/slot pilot](experiments/MASSIVE_INTENT_SLOT_PILOT.md): a local EN/JA structured-output diagnostic with negative semantic results.
- [JECS text-only pilot](experiments/JECS_CODE_SWITCH_PILOT.md): a narrow Japanese/English/code-switch byte-loss ablation on acted/read transcripts, not spontaneous conversation.

## Next development sequence

1. Author a new task-matched English/Japanese/code-switch development set independently of the excluded 101-case draft, obtain two semantic reviews plus qualified Japanese/bilingual review, and freeze it before quality-bearing comparisons.
2. Close rights, source-fit, and distribution reviews for task-matched training data; keep unapproved data and all final-holdout material out of training, tokenizer fitting, filtering, and prompts.
3. Compare task-focused formulations, tokenizers, sequence architectures, and training methods with controlled, compute-matched level-2 experiments after the reviewed development set is ready.
4. Measure the full system's behavior and user-visible latency, cold start, memory, unload, and wake-up behavior on representative CPU and GPU hardware.
5. Build and seal a statistically suitable final holdout for level-3 claims, then select a candidate only if it passes every behavioral and release gate.

No production model or ZenStream integration is implied by this research scaffold.
