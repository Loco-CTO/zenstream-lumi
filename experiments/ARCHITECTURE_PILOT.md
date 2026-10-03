# Model formulation and architecture pilot

**Status:** Planned and blocked. No model, tokenizer, or architecture has been trained, benchmarked, or selected. The approved training corpus is empty; the 101-case development draft has no ready cases or independent reviews; and there is no final holdout. The account/service terms for those Codex-generated cases, the intended model-weight distribution, the selected pilot tokenizer, and the target hardware inventory are unresolved. This protocol records a future controlled comparison; it is not evidence of a Lumi result.

## Question and scope

Which from-random-initialization model formulation and short-context sequence architecture can meet Lumi's reviewed English, natural Japanese, code-switch, action-safety, and response-quality requirements at the lowest measured training and deployment cost?

The pilot is staged to avoid conflating two questions:

1. **Formulation screen:** compare how a model maps reviewed turns and trusted context to Lumi's bounded structured response.
2. **Sequence-architecture screen:** within a formulation that remains viable, compare an all-attention control with one short-convolution/attention hybrid.

This is a research and model-comparison protocol only. It does not authorize admitting data, opening the sealed final holdout, training a release candidate, selecting a runtime, or integrating Lumi into ZenStream.

## Evidence basis and limits

Published results motivate candidates but do not predict a Lumi winner:

- The 2025 [RedLLM/DecLLM arXiv study](https://arxiv.org/abs/2510.26622) compares prefix-LM encoder-decoder and causal decoder-only models at roughly 150M–8B parameters, trained on 1.6T RedPajama tokens and instruction-tuned on FLAN. It reports better pretraining compute efficiency for decoder-only models and comparable or stronger downstream results with inference-efficiency advantages for encoder-decoder models after tuning. Its scale, data, training budget, and task mix are far from Lumi's planned pilot.
- The first-party [LFM2 technical report](https://arxiv.org/abs/2511.23404) reports up to 2x CPU prefill and decode speed for a gated short-convolution/grouped-query-attention design found through hardware-in-the-loop search. Its models were pretrained on 10–12T tokens. This motivates measuring a simple hybrid; neither its weights nor its speed claims transfer to Lumi.
- [SSM-Scope](https://arxiv.org/abs/2507.12442), published at ISPASS 2026, reports Transformers faster below 8K tokens and SSM advantages at much longer contexts on tested consumer and embedded GPUs. It does not measure CPU inference or random-initialized Lumi models. Lumi should use its reviewed context distribution before spending compute on a recurrent candidate.
- The single-author [Daedalus-150M preprint](https://arxiv.org/abs/2608.20210) reports a matched, from-scratch hybrid/all-attention comparison, with decode speed gains that grow with context length and are near zero at empty context. It is one general-task result and is not independently replicated; its preregistered matched-control design is useful, while its measured win remains only that author's result.

**Lumi evidence:** none. The statements above are published or author-reported findings, not Lumi measurements. The claims that task-focused outputs, short convolution, or an encoder-decoder formulation will help Lumi remain hypotheses until measured on approved data and reviewed development cases.

## Entry gates

Do not begin a quality-bearing pilot until every gate below is satisfied:

1. **Permitted data and use.** Every training example, response, tokenizer-training document, and derived artifact has a source, revision, content hash, and use scope that explicitly permits the planned training and intended model distribution. The provenance validator passes. Resolve the account/service terms and authorized use scope before using any of the existing 101 Codex-generated cases; otherwise exclude their text, labels, and derivatives from training, tokenizer fitting, evaluation, filtering, and prompt construction.
2. **Independent review.** The development set contains enough frozen, semantically diverse, ready cases for the English, Japanese, English-matrix code-switch, and Japanese-matrix code-switch slices, including no-action, false-action risks, arguments, clarification, multi-turn references, and response-language quality. At least two semantic reviewers label cases independently; a separate qualified Japanese/bilingual reviewer assesses Japanese meaning and naturalness. Labels and disagreements are reconciled under the annotation process. Reviewer availability alone does not count as completed review.
3. **Sequestered holdout.** A separately authored, source-separated final holdout has completed the required independent review, has frozen hashes and provenance, and is access-controlled. It is not loaded, scored, used for vocabulary fitting, inspected for failures, or used for stopping or hyperparameter selection during either pilot screen.
4. **Tokenizer control.** The tokenizer pilot has produced a provenance-approved training-only tokenizer candidate. Pin its artifact hash, normalization, vocabulary, and code version; use that exact tokenizer for every formulation and architecture candidate. Tokenizer changes require a separate experiment and a follow-up interaction comparison.
5. **Distribution and compute plan.** Record the intended model-weight distribution, legal review status, target CPU/GPU classes, available training device, memory limits, package budget, and fixed per-screen compute ceiling before training. A missing GPU does not block CPU-only research. Do not infer hardware or capacity from published device results.
6. **Implementation and evaluator frozen.** Pin all training and inference code, runtime/dependencies, parameter-count rules, scorer and case versions, sampling/decoding settings, numerical precision, stopping rules, and the experiment registration before generating candidate outputs.

The current inventory fails gates 1–5. A debug forward/backward pass may be used later to validate code paths, but it must be labeled as a smoke check and must not be scored or described as a model-quality comparison.

## Stage A: formulation screen

Use the same tokenizer, reviewed task examples, trusted-context projection, output schema, and training sources for each candidate. Every parameter tensor starts from a recorded random seed. No pretrained models, weights, embeddings, checkpoints, adapters, or unapproved teacher outputs are allowed. Any synthetic or model-generated training example must separately pass the synthetic-data provenance, permission, and review gates.

| Candidate | Formulation | Required control |
|---|---|---|
| A1 | Decoder-only causal Transformer that emits the full bounded structured response. | Standard all-attention reference with the project-approved serialization and fixed decode budget. |
| A2 | Encoder-decoder Transformer that encodes the input/context and generates the same response schema. | Count encoder and decoder parameters and FLOPs; apply identical output and stop rules. |
| A3, conditional | Direct structured prediction for finite action/clarification labels and typed slots, with explicit copy/span handling for open text and a separately scored concise response path. | Include only if the frozen capability contract makes its outputs complete and comparable; do not silently drop natural Japanese response requirements or open-ended constraints to make this candidate easier. |

A3 is a lower-cost hypothesis, not a preferred design. If its response pathway cannot preserve the same language, meaning, relevance, and concision contract, record it as not comparable and retain the evidence for rejection. A deterministic rules baseline may be measured as a cost reference, but is not a neural candidate and cannot replace the intended language-robustness target.

Before training, publish a small, executable example of each candidate's exact input and output representation and verify that every candidate can express every capability included in the evaluation. Unrepresentable outputs are a candidate failure, not a reason to relabel or remove evaluation cases.

## Stage B: sequence-architecture screen

For each formulation that survives Stage A, compare no more than these preregistered families at the first pass:

1. An all-attention Transformer control using the same formulation.
2. One simple, locally causal convolution/attention hybrid, with its exact layer schedule, convolution kernel, attention pattern, and parameter budget frozen before scoring. Implement from public architectural descriptions or independently licensed code; never reuse external weights.

Do not sweep many hybrid schedules in the first pass. A Mamba/SSM, RWKV, or other recurrent candidate enters a later declared experiment only if the reviewed development workload shows substantial long-context/state-retention cases or the first-pass measurements expose an attention-context bottleneck. Do not use an 8K threshold as a Lumi assumption; derive the decision from the observed, approved request and dialogue-length distribution.

## Matching and run procedure

For each screen, declare the candidate matrix, excluded variants, compute ceiling, stopping rule, and analysis before training. Run one tiny initialization/serialization smoke check first; it is only an implementation check. Then use paired random seeds and report every run.

Report all of these comparison views separately:

- **Compute matched (primary):** equal profiled model-training FLOPs under the same predeclared ceiling, including forward/backward passes, optimizer work, and auxiliary losses where measurable, plus wall time and device utilization. Report tokenizer build/preprocessing time and inference cost separately.
- **Source exposure matched:** each candidate receives the same ordered, provenance-approved source examples and response examples. Report token counts, source bytes/documents, updates, padding, measured FLOPs, and time; do not hide architecture-specific extra cost.
- **Parameter matched (secondary):** compare candidates within a preregistered tolerance in total trainable parameters, including embeddings, output projections, encoder/decoder components, and prediction heads. Report any unmatched parameters explicitly.

Use the same tokenizer revision, source-example ordering, approved data mix, train/development split, optimizer family and tuning allowance, update/early-stop rules, and paired seed list. If a candidate requires a different objective, report the objective and loss-token denominator as a confound; do not claim architecture-only causality. Do not let one candidate receive extra tuning, data, or training tokens without a separately named ablation.

One seed can reject an implementation bug or expose a catastrophic failure, but it cannot select a winner. Use at least three paired seeds for decision-grade comparisons; if the compute ceiling permits only a smaller screening run, label the result exploratory and repeat finalists with the preregistered seed count before choosing. Report dispersion and confidence intervals alongside pooled counts.

## Evaluation and deployment measurements

Only the frozen development set may guide candidate selection. Apply the fixed minimum thresholds in [EVALUATION_PLAN.md](../EVALUATION_PLAN.md) without relaxation: structured validity, simple action selection, argument extraction, negation/no-action, English, Japanese, both code-switch directions, multi-turn reference resolution, and difficult ambiguity. Report raw numerator/denominator, confidence bounds, and every required slice. Report false actions separately from missed actions; include unnecessary read-only calls as false actions. Keep state-changing and unclassified-action measures marked `not_measured` until the capability registry is approved.

For each candidate, retain raw outputs and record:

- Parameter count, tokenizer/model artifact bytes, chosen precision/quantization, and exact build revision.
- Cold process start, model load, first useful output, input/prefill latency, decode latency, and full-request p50/p95 over the actual approved request and multi-turn context distribution.
- CPU-only latency and peak/resident RAM on each declared low-resource, home-server, and desktop CPU class. Test GPU only on declared practical devices and report it separately; it is optional and earns no assumed advantage.
- Idle CPU/GPU activity and memory with Lumi disabled and enabled-but-idle, plus unload-to-idle and wake/reload latency. Use repeated runs with pinned thread counts, power/performance mode, and runtime settings.
- The inference runtime and kernel/backend for every result. First compare candidates in a common reference implementation where available; also measure optimized deployment backends as a separate end-to-end result. A backend difference must not be attributed to architecture alone.

Architecture screening is not the final release benchmark. Re-run finalists, after model and runtime choices are stable, across all required device classes and verify the absent/disabled installation boundary in ZenStream.

## Decision and record

Reject a candidate if it cannot express the frozen contract, violates a quality threshold, has a safety-critical slice regression, exceeds a predeclared resource ceiling, or depends on an unapproved artifact. Among candidates that pass every behavioral gate, compare package size, cold start, p95 useful-response latency, peak RAM/VRAM, idle activity, unload/wake behavior, and reproducibility. Select only from measured evidence; do not trade away a product threshold to win a resource metric.

Use [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for every run. Preserve the full registration, environment, hardware identifiers, source and tokenizer manifest hashes, initialization seeds, code/dependency revisions, exact FLOPs/tokens/examples, raw predictions, scorer output, logs, and run outcome. Record each rejected and inconclusive candidate and why. Do not publish a selected architecture until the result is repeated with the required seeds and then confirmed once on the still-sealed final holdout under its separately approved evaluation procedure.

No run has started. The families listed above are experimental candidates only; no formulation or architecture winner, parameter size, compute budget, or target runtime has been selected.
