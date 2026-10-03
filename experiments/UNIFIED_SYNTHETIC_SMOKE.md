# Unified synthetic random-init smoke

**Date:** 2026-10-03
**Evidence level:** 1 — exploratory
**Decision:** reject this byte-GRU formulation as a usable unified candidate; retain the run as an end-to-end failure diagnostic.

## Question and scope

Can one small model trained from random initialization learn a complete serialized ZenStream decision, action, arguments, clarification flag, presentation intent, and concise user-facing message, then save, reload, and run on CPU?

The training packet was newly authored as fixed English, Japanese, and English/Japanese examples inspired by the current Lumi objective. It covers harmless conversation, acknowledgement, media comments, supported playback/search, a negated action, ambiguous commands/references, noisy insufficient-understanding inputs, capability questions, and unsupported requests. Fictional titles and constraints are used. No synthetic example was produced by an LLM teacher.

This is a single exploratory engineering run, not a formal evaluation set, development-quality comparison, or language-quality claim. Its 42 exploratory development examples have family IDs disjoint from the 42 training examples, but both splits were written by the same author and generator. Neither split has independent semantic or qualified Japanese/bilingual review. The examples were not added to evaluation cases or the draft inventory.

## Provenance

- Goal source: current user-provided objective attachment, SHA-256 `sha256:fc1bc61d354a9534a7a6692646c52d42c2cb2c5a5413025f0b48fbe461e5232c`.
- Historical template-builder source SHA-256: sha256:3bbfb41e068e75ed57c450dc371c94598e33020d70ba673d110f4f71b78f3ef3. The original builder contained the fixed examples but is not included in the current tracked source tree; the current runner loads only hash-pinned local JSONL and contains no prompt or gold text.
- Reused model code: [`random_init_conversation_response_smoke.py`](random_init_conversation_response_smoke.py), pinned SHA-256 `sha256:6cb2a3c8d1f56e7c304194db40132eef3b34f39d505f77e53b4385ffba1cf780` at base commit `039b54310921a7aa32cdacd505f8fa7f1391db2e`.
- Train: 42 examples, 21 families; SHA-256 `sha256:1a28bb3abc9be316d46f6e3e0c7cced4e55a16d7f9fe3512818a725626d15247`.
- Exploratory development: 42 examples, 21 families; SHA-256 `sha256:88c7cd5614f589543838ccb15b545821b320de9397e33eaaabe4ffec4bc306fe`.
- Combined generated packet: 84 examples; SHA-256 `sha256:02790c733a85a76bb4be63a3f23413b1bbe13e6bdd06272ef77ded0e2e5bdaca`.
- The local source manifest is retained at .lumi-data/lumi-unified-synthetic-smoke-20261003-a2/source-manifest.json, SHA-256 sha256:b9cd6d28fa38ad434a13a03d9c5e2c26c80f0f3b46a6482dae574b0cb7a1f61d.
- Frozen train/dev JSONL and generated samples, report, predictions, and weights remain outside Git under the controlled .lumi-data run directory. The tracked reproduction runner requires those exact inputs by SHA-256 and does not embed or regenerate the examples. The synthetic generation record captures the local scope and packet checksum; distribution approval is false.
- The excluded 101-case Codex draft and all derivatives were not opened, loaded, tokenized, derived from, or evaluated on. JECS, JMultiWOZ, MASSIVE, pretrained weights, and teacher outputs were not used.

## Model and training

The model is the existing single-layer tanh recurrent byte-level encoder-decoder: a shared 24-dimensional UTF-8 byte embedding, 32-dimensional hidden state, and autoregressive decoder. Its one target sequence is canonical JSON containing the full structured object and message. The model has 18,321 parameters and starts from NumPy Gaussian random weights with seed 1729. There are no pretrained assets.

Training used Adam at 0.01 for 64 epochs and 2,688 example updates, with single-threaded NumPy CPU execution and gradient norm clipping at 1.0. Mean training byte cross-entropy fell from **5.55810** to **0.16032** in **23.54 seconds**. The compressed weights are **70,433 bytes** (SHA-256 `sha256:8d086539724572cdef98da67ab4d9e6deb20b4719389503aa8051ef603b9cf22`). Reload preserved the exact parameter-state hash `sha256:36dc5e761213d57483d50449bba1b629f4b7259c90860e54b2dbeb90fcf32f6a`; loading took **1.01 ms**.

## Results

| Slice | Examples | Syntactically valid JSON | Full output schema valid | Exact combined output |
|---|---:|---:|---:|---:|
| English | 24 | 24/24 | 0/24 | 0/24 |
| Japanese | 12 | 12/12 | 0/12 | 0/12 |
| English/Japanese mixed | 6 | 6/6 | 0/6 | 0/6 |
| **All exploratory development** | **42** | **42/42** | **0/42** | **0/42** |

The model produced the same output for all 42 development inputs and all 42 training replays:

```json
{"action":"playback_handoff","requires_clarification":false}
```

This parses as JSON, but it is not a valid Lumi output: it places a presentation-intent value in `action` and omits `decision`, `arguments`, `message`, and `presentation_intent`. EOS fired after the same **60-byte** output despite a **512-byte** generation limit; the mean gold target was **185.62 bytes**. Exact decision, action, argument, clarification, and message accuracy were each **0/42** for development and train replay. The false-action rate is **unscored**, because the schema-invalid object has no `decision` field; it must not be read as zero false actions.

The observable failure is the same incomplete 60-byte JSON object for every input, followed by EOS. This is free-running mode collapse with early EOS; it is not a literal prefix of the canonical targets. The low teacher-forced byte loss did not translate into complete sequence generation. This run alone cannot distinguish exposure bias from the small recurrent bottleneck or the serialized-object objective as the cause. The complete metrics, outputs, loss history, runtime measurements, and artifacts are retained in the local `report.json`.

CPU resource measurements include the Python and NumPy process: first development generation **0.74 ms**, warm p50 **0.77 ms**, warm p95 **1.06 ms**; working set was **37,150,720 bytes** before reload, **37,171,200 bytes** after reload, and **44,163,072 bytes** peak. These figures are one Windows CPU run, not model-only memory or production benchmarks. GPU was unused.

## Interpretation and next seam

This run establishes that one randomly initialized CPU model can be trained, serialized, reloaded, and invoked with combined JSON targets. It **fails the structured-output gate even on its training examples**, so it does not demonstrate a working unified Lumi path. No behavior, language quality, safety, grounding, or product threshold is met or claimed.

The failure seam is the unrestricted byte-level `fit()`/`generate()` path in the reused response-smoke engine: semantic structure, arguments, and free text all compete in one long autoregressive byte target, and greedy inference terminates at a repeated incomplete object. A next controlled variant should use one shared input encoder with typed decision/action/argument heads and a response decoder conditioned on those predicted semantics, followed by canonical serialization and the existing output-schema check. That would keep combined-system behavior measurable while making false-action and argument metrics scorable. This report does not implement or train that variant.

## Reproduction

Use the Python 3.12 environment with NumPy 2.3.5. Choose a new empty output directory below a controlled `.lumi-data` directory; the runner refuses output paths inside a Git worktree and does not accept an arbitrary dataset.

```powershell
$Python = 'C:\path\to\python.exe'
$Dataset = 'C:\path\outside\git\.lumi-data\lumi-unified-synthetic-smoke-20261003-a2'
$Run = 'C:\path\outside\git\.lumi-data\lumi-unified-synthetic-smoke-rerun'
& $Python experiments\random_init_unified_synthetic_smoke.py --dataset-dir $Dataset --output-dir $Run
```

The runner pins seed 1729 and the reused engine source hash. It verifies the frozen train/dev JSONL SHA-256 values before training, then writes packet copies, manifests, model, replay/development predictions, and report into $Run. It refuses arbitrary data and any source or output path inside a Git worktree.
