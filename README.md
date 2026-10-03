# Lumi

Lumi is a proposed optional, local natural-language capability for ZenStream. It is developed and released independently in this repository. ZenStream must remain fully useful when Lumi is absent, disabled, or unavailable.

## Project status

This repository currently contains the research, evaluation, and provenance foundation only. No Lumi model, inference service, training pipeline, approved training corpus, or runtime dependency has been created or selected. A 101-case draft evaluation inventory exists in local controlled storage, but no case is ready or approved and its text is not checked into this repository. No pretrained model weights are permitted for Lumi's runtime model; any eventual Lumi model must start from random initialization.

Architecture, tokenizer, model size, runtime, training corpus, integration protocol, and artifact licensing remain open research decisions. Research claims are not Lumi measurements. See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the initial evidence review and its limitations.

## Product boundaries

- Lumi is opt-in. A disabled installation must not download a model, start an inference process, reserve meaningful RAM or VRAM, or create AI-specific CPU/GPU work.
- Lumi interprets language and can suggest structured intent. ZenStream remains authoritative for media, permissions, playback, metadata, user state, and all state-changing validation.
- Lumi must not emit executable application code, SQL, shell commands, or arbitrary system commands as part of normal interaction.
- CPU inference is required. GPU use is optional and must show a measurable benefit on supported hardware.
- Initial language targets are English, Japanese, and English/Japanese code-switching. Japanese evaluation must be natural Japanese and reviewed by competent speakers.
- Data provenance, licensing, benchmark integrity, and reproducibility are release requirements.

## Project records

- [Evaluation plan](EVALUATION_PLAN.md): target thresholds, error categories, measurement, and holdout rules.
- [Evaluation tools](evaluation/README.md): canonical case/output/prediction formats, a hash-bound independent-review ledger, a blind loopback review workbench and draft-only ledger importer, a candidate-input projection, provenance/review-gated runner, coverage audit, and standard-library scorer; the controlled draft inventory has no human review records or ready cases yet.
- [Annotation guide](evaluation/ANNOTATION_GUIDE.md): draft labeling, response contracts and presentation intent, the controlled review ledger, Japanese/code-switch review, split grouping, and v4 coverage limits.
- [Research notes](RESEARCH_NOTES.md): current findings, limitations, open hypotheses, and proposed experiments.
- [Research references](RESEARCH_REFERENCES.md): sources that influence design; these are not training data.
- [Tokenizer pilot protocol](experiments/TOKENIZER_PILOT.md): a controlled EN/JA/code-switch tokenizer comparison, gated on approved text and reviewed development cases.
- [Data sources](DATA_SOURCES.md): inclusion policy and human-readable provenance summary.
- [Machine-readable provenance](provenance/data_sources.json): currently empty because no external data has been approved or used; [the validator](provenance/README.md#provenance-gate) checks source/sample joins and split-safe evaluation hashes.
- [Synthetic-data provenance](provenance/synthetic_data.json): currently empty because no synthetic data has been generated or used.
- [Model card](MODEL_CARD.md): pre-release placeholder; no model exists yet.
- [Third-party notices](THIRD_PARTY_NOTICES.md): current code license and outstanding artifact-license research.
- [Experiment record template](experiments/EXPERIMENT_TEMPLATE.md): reproducibility and comparison fields.

## Initial sequence

1. Build and review a semantically diverse English/Japanese/code-switch evaluation set, with a sequestered final holdout and expert review for Japanese.
2. Audit candidate public and synthetic data for provenance, permitted use, and contamination risk before any dataset is admitted.
3. Compare task-focused model formulations, tokenizer candidates, and sequence architectures using small, compute-matched experiments.
4. Train from random initialization only after the evaluation gates and data manifests are ready.
5. Benchmark model quality and user-visible latency, cold start, memory, unload, and wake-up behavior on each target hardware class.
6. Select the smallest candidate that passes the behavioral targets and has the best measured product experience; retain evidence and rejected alternatives.

No production model or ZenStream integration is implied by this research scaffold.
