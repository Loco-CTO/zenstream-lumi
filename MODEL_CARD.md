# Lumi model card

**Status:** No model trained or released. This is a placeholder and makes no capability or performance claim.

## Intended use

When a candidate passes the evaluation gates, Lumi is intended to interpret optional English, Japanese, and English/Japanese code-switched natural-language requests for ZenStream and return bounded intent information. ZenStream remains authoritative for library facts, permissions, user state, and actions.

## Out of scope

- General-purpose factual assistant or authoritative media catalog.
- Unvalidated state-changing actions.
- Executable code, SQL, shell commands, or arbitrary system commands.
- Any mandatory dependency in core ZenStream behavior.

## Model and training

- Model version: Not established.
- Architecture and tokenizer: Not selected.
- Parameter count and model file size: No model exists.
- Initialization: Any eventual Lumi runtime model must start from random initialization; pretrained checkpoints and derived adapters are prohibited.
- Training stages and token counts: Not established; no training has occurred.
- Training data: None. See [DATA_SOURCES.md](DATA_SOURCES.md) and `provenance/data_sources.json`.
- English/Japanese data composition: Not established.

## Evaluation and limitations

- Behavioral evaluation: Not run.
- English, Japanese, code-switching, action reliability, grounding, and structured-output results: None.
- Known weaknesses: Not yet measured. Do not use this placeholder as evidence of readiness.
- Hallucination/unsupported-claim rate: Not measured.

## Deployment

- CPU-only support is a project requirement, not yet implemented or verified.
- GPU support, quantization, runtime formats, RAM/VRAM, latency, cold start, idle behavior, and unload behavior: Not established.
- Lumi must be opt-in and must not load or download model artifacts when disabled.

## License

Model-weight and training-artifact licensing have not been decided. Do not assume they inherit the repository's current code license; research and document obligations before release.
