# Lumi research notes

- Research snapshot: 2026-10-02
- Status: Initial literature and tooling scan; no candidate architecture or runtime selected.
- Lumi experiments: None. Every performance or quality statement below is attributed to its source and is not a Lumi result.

## Workload and evidence boundary

Lumi's intended workload is short, contextual media intent interpretation with a bounded machine-readable response. It is narrower than open-ended chat but includes difficult language phenomena: negation, hypotheticals, indirect requests, omissions, slang, typos, Japanese casual speech, code-switching, follow-ups, and ambiguity. The decisive metrics are therefore action/no-action correctness, argument extraction, grounding, and response reliability, alongside latency and resource use.

**Hypothesis (untested):** A task-focused model and small output space may meet the product targets with fewer parameters and fewer generated tokens than a general-purpose chat model. It must be compared against other model formulations; this is not a decision to use a classifier, encoder, decoder, or a particular output protocol.

## Multilingual training and data mixture

### Published findings

- Google's 2026 [ATLAS work](https://research.google/blog/atlas-practical-scaling-laws-for-multilingual-models/) reports 774 multilingual training runs from 10M to 8B parameters, spanning 400+ languages. It fits scaling and transfer rules for model size, data amount, and language mixtures. Its public summary also reports a multilingual compute-efficiency cost, particularly for English. This supports doing measured English/Japanese data-mixture sweeps instead of assuming equal proportions or adding languages without evidence. The study optimizes vocabulary-insensitive loss over general language data; it does not establish the best mixture for Lumi's intent task or code-switch distribution.
- The 2025 ACL Findings paper on [code-switching curriculum learning](https://aclanthology.org/2025.findings-acl.407/) reports staged token-level switching, sentence-level switching, and monolingual training, and includes experiments involving Japanese. Its base models are already pretrained (Qwen 2, Gemma 2, and Phi 3.5), so the result does not show that this curriculum transfers unchanged to random-initialization training.
- The ACL 2026 [*Lost in the Mix*](https://aclanthology.org/2026.acl-long.2080/) study constructs linguistically grounded code-switch variants of comprehension benchmarks. It reports that code-switching can alter model accuracy and that fine-tuning provides more consistent mitigation than in-context cues. Its benchmark task and model scale differ from Lumi's media requests; use it to inform evaluation design, not as a proxy for EN/JP command accuracy.
- [*MiniLingua*](https://arxiv.org/abs/2512.13298) reports a 1B model trained from scratch for 13 European languages. It is a useful multilingual scratch-training case study, but it does not establish a suitable parameter count, data mixture, or Japanese capability for Lumi.
- The 2025 Basque sub-1B study ([Urbizu et al.](https://aclanthology.org/2025.mrl-main.35/)) reports that, with roughly 500M words of target-language data, continual pretraining a multilingual checkpoint substantially outperformed scratch training. This is a warning about the data-efficiency challenge of scratch training, not an allowed recipe for Lumi. Basque resource conditions and the evaluated tasks are not directly transferable to English/Japanese.

### Lumi hypotheses and next experiments

1. Construct an evaluation slice stratified by English, natural Japanese, and naturally code-switched English/Japanese, including the actual proportions expected in product use.
2. At a fixed token and compute budget, compare at least two defensible EN/JA mixtures and a broader multilingual mixture. Score behavioral metrics as well as held-out language loss; do not select a mixture on loss alone.
3. Once a from-scratch baseline exists, test code-switch curriculum schedules as controlled ablations. Preserve a matched non-curriculum control and prevent evaluation examples from entering generated or training data.

## Japanese and code-switch tokenization

NTT's [technical reporting on Tsuzumi](https://www.ntt-review.jp/archive/ntttechnical.php?contents=ntr202608fa2_s.html) describes purpose-built Japanese tokenization and filtering for Japanese text quality; these are first-party claims, not an independent Lumi comparison. A 2026 tokenizer study ([Lee et al.](https://arxiv.org/abs/2606.15044)) compares tokenizers across 11 Southeast Asian languages and trains controlled 1.5B models, but does not include Japanese. Together these motivate measuring Japanese token fertility, byte coverage, vocabulary parameter cost, and downstream behavior on licensed English, Japanese, and code-switch samples. Neither source settles Lumi's tokenizer design or scale.

**Open questions:** Compare a jointly trained multilingual subword tokenizer with byte/character-safe or other plausible alternatives; measure common Japanese forms, kanji/kana variation, romanized and English titles, punctuation, emojis, and mixed-script boundaries. Include the tokenizer's training corpus and revision in the data provenance manifest. Avoid vocabulary growth that improves average token counts but harms tail text or consumes too many embedding parameters.

## Model architecture candidates

No family is selected. Candidate comparisons must use the same approved data, comparable parameter/compute budgets, equivalent training opportunities, and Lumi's behavioral evaluation.

| Family to investigate | Published evidence relevant to Lumi | Main limitation to test |
|---|---|---|
| Transformer with softmax attention | Mature training and inference ecosystem; direct baseline for language and context handling. | KV memory grows with context; short prompts may not benefit from more complex sequence kernels. |
| Mamba-2 and Transformer/SSM hybrids | A controlled 8B comparison ([Waleffe et al.](https://arxiv.org/abs/2406.07887)) reports that pure SSMs lag on copying and some in-context tasks, while its Mamba-2 hybrid outperformed its matched Transformer on evaluated tasks and was predicted to generate faster. | The comparison is at 8B and very large token budgets; it does not establish small CPU results. Contextual reference resolution and exact title copying are important Lumi probes. |
| RWKV-7 | The 2025 report ([Peng et al.](https://arxiv.org/abs/2503.14456)) includes multilingual models from 0.19B to 2.9B and describes constant-size recurrent inference state with parallelizable training. | Published language-model results and implementation needs must be rechecked for Lumi's task, target CPUs, quality at the lowest scale, and maintainability. |
| Gated DeltaNet and hybrids | The ICLR 2025 work ([Yang et al.](https://arxiv.org/abs/2412.06464)) reports improvements over Mamba-2 and DeltaNet on several tasks and introduces sliding-window-attention/Mamba hybrids. | Kernel support and ecosystem portability may be GPU-centric; benchmark actual CPU-only decode and short-prompt latency. |
| Task-focused encoder, classifier, or structured prediction | Could avoid long free-form generation for finite actions and slots. | Must handle multi-turn references and open-ended media constraints without brittle labels; research and benchmark before adoption. |

The Mamba comparison, RWKV-7, and Gated DeltaNet are evidence that alternatives deserve a fair small-scale comparison, not evidence that one will win. A non-neural deterministic baseline is also useful as a lower-cost reference, not as Lumi's chosen model.

## Structured responses and action safety

JSONSchemaBench (2025, [Geng et al.](https://arxiv.org/abs/2501.10868)) evaluates constrained-generation frameworks on schema validity, constraint coverage, efficiency, and output quality. Constrained decoding can limit output syntax, but syntactic validity alone cannot prove correct intent, slots, or safe action selection. Lumi evaluation must score schema validity separately from semantic accuracy, and ZenStream must validate each proposed operation against current permissions and state.

**Hypothesis (untested):** A compact, versioned response schema with constrained decoding and server-side allow-list validation may reduce malformed responses and contain model errors. The schema, decoder, and response format remain open decisions until candidate runtimes are compared.

## Inference, quantization, and resource behavior

- The current [`llama.cpp` project](https://github.com/ggml-org/llama.cpp) documents CPU execution and multiple accelerator backends, including CUDA, HIP, Metal, SYCL, and Vulkan. This breadth makes it a useful deployment candidate, but model-format support, Japanese tokenization, load/unload behavior, and real hardware performance still need testing.
- [ONNX Runtime GenAI](https://onnxruntime.ai/docs/genai/howto/install.html) documents separate CPU, CUDA, and DirectML packages. This provides another portable deployment candidate; export constraints, platform coverage, startup cost, and supported model operations need testing.
- Current [TorchAO documentation](https://docs.pytorch.org/ao/stable/workflows/inference.html) includes CPU-oriented x86 int4-weight/int8-activation quantization options. An August 2026 paper ([Xia et al.](https://arxiv.org/abs/2608.18182)) reports INT8 throughput gains for BERT-family models on Intel Xeon CPUs. Those results are not measurements for autoregressive SLM generation or low-resource home-server CPUs and must not be transferred to Lumi without benchmarking.
- The inference comparison must record cold start, warm short-command latency, time to first useful output, CPU/GPU throughput, peak and idle RAM/VRAM, idle utilization, unload time, and wake/reload time. Compare CPU-only against GPU backends; faster generation is not sufficient if the model stays resident or takes too long to load.

## Data quality, synthetic data, and licensing

Recent ACL work on [multilingual synthetic pretraining data](https://aclanthology.org/2026.acl-long.1002/) reports that quality filtering can help under its evaluated language and task conditions; its translation method can trade semantic fluency for knowledge injection. Separate ACL 2026 work on [synthetic-data diversity](https://aclanthology.org/2026.findings-acl.360/) finds that multiple generator sources can mitigate output-distribution collapse in its fine-tuning setup, while synthetic fine-tuning can also remove safeguards. A 2026 function-calling data-generation paper ([GenesisFunc](https://aclanthology.org/2026.acl-long.1319/)) identifies weak diversity and quality control as common problems and proposes multi-stage generation checks; its reported student is an 8B model fine-tuned on generated data. These results argue for measuring diversity and label correctness, but do not establish a reliable Lumi data-generation recipe: the studies use different domains, model families, languages, and tasks.

**Hypothesis (untested):** Carefully validated teacher-generated examples may efficiently cover rare negation, ambiguity, correction, and code-switch cases. Test a small, manually audited set first; count semantic families, preserve hard negatives, compare multiple generators if permitted, and reject data that harms human-written slices or safety-critical behavior.

No external corpus, benchmark, tokenizer corpus, synthetic example, model weight, or training artifact has been admitted to Lumi. References in this document inform research only; they are not training sources. Before use, each source needs a recorded version/revision, access date, license and permissions, transformations, filters, deduplication, split use, and measured contribution. Synthetic examples require generator provenance and automated plus human validation. Unknown or unclear rights mean exclusion until resolved.

The training-data manifest is currently empty. A future training run must consume a frozen manifest and preserve the exact manifest and data-processing revisions used for that run. Development evaluation and final holdout material must remain separate from every training, generation, and filtering input.

## Research areas not yet assessed

These remain open; no method below has been accepted or rejected for Lumi, and no Lumi experiment has been performed.

| Area | Initial relevance to Lumi | Main risk or uncertainty | Next evidence needed |
|---|---|---|---|
| Pruning, sparsity, low-rank methods, and parameter sharing | Could reduce model bytes, RAM, or compute. | Much compression evidence begins with pretrained models; benefits may not transfer to a from-scratch model or the available CPU kernels. | Research scratch-compatible methods, then compare against a matched dense candidate only after a behavioral baseline exists. |
| Knowledge distillation | Teacher-provided labels or preference critiques could improve task coverage without using teacher weights. | Teacher errors, provider terms, language imbalance, synthetic-data repetition, and holdout leakage. | Review current data-distillation evidence and rights; pilot a small audited set with a human-written control. |
| Retrieval assistance and prompt/context caching | ZenStream can provide authoritative, current library or conversation context without training Lumi to memorize user state. | Stale or overbroad context, privacy, cache invalidation, extra latency, and confusing tool failure with model uncertainty. | Research bounded retrieval and cache invalidation; compare no-context, authoritative-context, and failed-tool cases in evaluation. |
| Cold start, on-demand loading, unloading, and prompt-processing acceleration | Required to keep disabled/idle resource use negligible and short requests responsive. | Process startup, model load, compilation, and accelerator initialization can outweigh generation for short commands. | Measure process start, load, prefill, first useful output, resident idle use, release, and wake latency before selecting runtime lifecycle. |
| Speculative or assisted decoding | Could reduce generation latency on some hardware. | A small model's output is short; draft-model memory and verification overhead may cost more than they save. | Revisit only if representative short-request generation is a measured bottleneck. |
| Small-model reasoning/post-training methods | May help ambiguity, correction, and multi-turn reference resolution. | Extra reasoning tokens can increase latency and do not guarantee correct action intent. | Research task-specific methods and evaluate false actions plus latency, not only answer quality. |

## Decisions and claims ledger

### Fixed project requirements (from the user goal)

- Lumi's runtime model starts from random initialization; no pretrained weights, checkpoints, or derived adapters.
- Lumi is optional and independently removable; CPU-only inference is mandatory.
- English, Japanese, and English/Japanese code-switching are the initial formal targets.
- ZenStream owns authoritative application and library state and validates actions.
- Do not preselect architecture, tokenizer, runtime, model size, or training corpus.
- Initial behavior thresholds are listed in [EVALUATION_PLAN.md](EVALUATION_PLAN.md).

### Lumi experimental findings

None yet.

### Hypotheses awaiting measurement

- A task-focused model may be much smaller than a general-purpose conversational model while meeting action and slot targets.
- Some broader multilingual exposure may improve robustness, but can trade off with English/Japanese sample efficiency.
- Japanese-aware tokenization and explicit code-switch examples may improve target behavior enough to justify vocabulary/data cost.
- Constrained decoding and on-demand model loading may help reliability and idle resource goals, but need end-to-end measurement.

## Next research gates

1. Finalize a behavior taxonomy, annotation guide, development set, and sealed holdout before substantial model training.
2. Complete a rights and availability audit for candidate English and Japanese corpora and benchmarks; admit none by default.
3. Run tokenizer and small-model architecture pilots under a declared compute ceiling.
4. Review teacher-generated-data methods and their diversity, factual validation, licensing, and contamination risks before generating any data.
5. Compare runtime and quantization options on representative 4 GB, 8 GB, 16+ GB, and GPU systems.
