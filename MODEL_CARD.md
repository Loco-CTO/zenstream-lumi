# Lumi model card

**Status:** Small exploratory random-init structured-prediction and text-generation components have been trained and measured locally; neither is a complete Lumi candidate, and neither has been selected or released. This record makes no product capability or performance claim.

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
- English/Japanese data composition: No product training mix is established. One local JECS text-only byte-LM ablation used 984 training families and 235 public development families; the [pilot report](experiments/JECS_CODE_SWITCH_PILOT.md) documents the mixture and limits.

### Exploratory conversation response component

- Architecture: single-layer tanh recurrent byte-level encoder-decoder; it uses no learned tokenizer.
- Size: 18,321 parameters and a 69,972-byte local artifact. This is not a selected Lumi model size.
- Training: random initialization, 300 epochs, four illustrative user-provided examples, CPU only.
- Development behavior: generated valid UTF-8 for two illustrative prompts, but reused an unrelated response and gave a generic rather than task-specific clarification. No human review or quality score was performed.
- Scope: English only; not integrated with the structured predictor or ZenStream. Details are in [the conversation response smoke](experiments/CONVERSATION_RESPONSE_SMOKE.md).

### Exploratory JECS byte-LM ablation

- Architecture: random-initialized, one-layer tanh recurrent UTF-8 byte language model with fixed language tags and no learned tokenizer.
- Training: 18,086 parameters; one CPU-only, single-seed comparison of Japanese/English training rows against Japanese/English/code-switch rows from a small, text-only JECS v1 scope.
- Development result: the code-switch-augmented run lowered bits per UTF-8 byte on 96 held-out JECS code-switch rows; English was effectively unchanged and Japanese moved slightly. This is a public development result on acted/read text, not evidence of natural conversational ability or product quality.
- Weights: approximately 68 KB each, kept in controlled local storage. The source license and data lineage do not by themselves clear either artifact for public distribution.
- Details and exact metrics: [JECS pilot report](experiments/JECS_CODE_SWITCH_PILOT.md).

## Evaluation and limitations

- Development/release evaluation: Not run. The 5-example intent smoke and 2-example response smoke are too small, illustrative, and unreviewed for product claims.
- English, Japanese, code-switching, action reliability, grounding, and structured-output task results: None. JECS byte-level held-out text losses are recorded as a narrow language-model diagnostic; they are not task or response-quality results.
- Known prototype weaknesses: two false actions on future-intent examples, failure to extract one code-switched title, and no genre/year/runtime/watched-state extraction. These are five-example exploratory findings only; do not use them as population estimates or readiness evidence.
- Natural conversation, unsupported-request handling, and low-confidence behavior were not evaluated by the structured-prediction smoke. A classifier or slot extractor alone does not satisfy Lumi's user-facing conversation requirement.
- The response smoke emitted nonempty valid UTF-8 on both illustrative development prompts, but reused an unrelated training reply for a media comment and used generic uncertainty wording for an ambiguous playback request. This is a qualitative diagnostic from two unreviewed examples, not a score. It has no Japanese or code-switch evidence.
- Hallucination/unsupported-claim rate: Not measured.

## Deployment

- CPU-only support is a project requirement. Both short-lived NumPy prototypes ran on CPU; their process working-set measurements include Python and NumPy and do not establish production inference or lifecycle behavior.
- GPU support, quantization, runtime formats, RAM/VRAM, latency, cold start, idle behavior, and unload behavior: Not established.
- Lumi must be opt-in and must not load or download model artifacts when disabled.

## License

Model-weight and training-artifact licensing have not been decided. Do not assume they inherit the repository's current code license; research and document obligations before release.
