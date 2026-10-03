# Lumi model card

**Status:** A small exploratory random-init structured predictor has been trained and measured locally; no Lumi candidate has been selected or released. This record makes no product capability or performance claim.

## Intended use

When a candidate passes the evaluation gates, Lumi is intended to interpret optional English, Japanese, and English/Japanese code-switched requests, return bounded intent information where appropriate, and provide concise natural-language conversation, clarification, and graceful unsupported/uncertain responses. ZenStream remains authoritative for library facts, permissions, user state, and actions.

## Out of scope

- General-purpose factual assistant or authoritative media catalog.
- Unvalidated state-changing actions.
- Executable code, SQL, shell commands, or arbitrary system commands.
- Any mandatory dependency in core ZenStream behavior.

## Model and training

- Model version: No product model is established. The exploratory-only `lumi-random-init-intent-smoke-v1` is documented in [the experiment record](experiments/RANDOM_INIT_INTENT_SMOKE.md).
- Exploratory architecture: two independent one-hidden-layer MLP heads over fixed hashed Unicode character n-gram features. No learned tokenizer is used. Neither is selected for a Lumi candidate.
- Parameter count and model file size: The smoke prototype has 164,264 parameters and a 618,501-byte local artifact; these measurements do not describe a selected Lumi model.
- Initialization: Any eventual Lumi runtime model must start from random initialization; pretrained checkpoints and derived adapters are prohibited.
- Training stages and token counts: No product training plan or token count is established. One 120-epoch level-1 smoke run is recorded in the experiment note.
- Training data: No production-quality or release training data is admitted. The smoke used 12 local user-brief examples; see [DATA_SOURCES.md](DATA_SOURCES.md) and `provenance/data_sources.json`.
- English/Japanese data composition: Not established.

## Evaluation and limitations

- Development/release evaluation: Not run. The 5-example exploratory smoke result is too small and unreviewed for product claims.
- English, Japanese, code-switching, action reliability, grounding, and structured-output results: None.
- Known prototype weaknesses: two false actions on future-intent examples, failure to extract one code-switched title, and no genre/year/runtime/watched-state extraction. These are five-example exploratory findings only; do not use them as population estimates or readiness evidence.
- Natural conversation, unsupported-request handling, and low-confidence behavior were not evaluated by the structured-prediction smoke. A classifier or slot extractor alone does not satisfy Lumi's user-facing conversation requirement.
- Hallucination/unsupported-claim rate: Not measured.

## Deployment

- CPU-only support is a project requirement. A short-lived NumPy prototype ran on CPU, but no production inference runtime or lifecycle has been implemented or verified.
- GPU support, quantization, runtime formats, RAM/VRAM, latency, cold start, idle behavior, and unload behavior: Not established.
- Lumi must be opt-in and must not load or download model artifacts when disabled.

## License

Model-weight and training-artifact licensing have not been decided. Do not assume they inherit the repository's current code license; research and document obligations before release.
