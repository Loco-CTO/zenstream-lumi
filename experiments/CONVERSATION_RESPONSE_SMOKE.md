# Random-init conversation response smoke

**Date:** 2026-10-03

**Evidence level:** 1 — exploratory

**Decision:** retain as an engineering baseline only; reject as a Lumi candidate.

## Question

Can a small model trained entirely from random initialization condition on a user prompt and generate a bounded natural-language response, save and reload its weights, and run on CPU?

## Research basis

Byte-level modeling avoids a fixed learned subword vocabulary. ByT5 reports robustness to noisy text and operates directly on UTF-8 bytes, but its released model sizes start at 300 million parameters; this does not show that a tiny byte-level recurrent model will learn useful conversation. The newer Autoregressive U-Net work pools bytes into word and larger scales and studies 25 million to 500 million parameter models, a more relevant architectural alternative for future matched experiments but still far beyond this smoke's data and scale. The Japanese-focused PLaMo report describes scratch training on two trillion tokens, underscoring that byte coverage alone is not Japanese language understanding. Sources and limits are recorded in [RESEARCH_REFERENCES.md](../RESEARCH_REFERENCES.md).

The single-layer recurrent encoder-decoder was selected only as the smallest practical CPU experiment for conditional text generation. This is not evidence for choosing a recurrent architecture over a Transformer, SSM, hybrid, retrieval, or structured-predictor design.

## Data and provenance

- Source: the user-provided updated Lumi goal attachment, SHA-256 e7adbedb03c9b1adc1b9939913e7048ea0d27be61e029b9d13814fc4d33bfc28.
- Scope: local exploratory training and a tiny development-only smoke for this requested project; no release or model distribution approval.
- Local dataset: 4 training and 2 development prompt/response examples, divided by semantic family. The examples are illustrative, not a benchmark or a production response template.
- Dataset SHA-256: 974d45ea0a2b650a5246fbbaeea5f2d4edf0c1db933a511041456218517acaaa.
- Local manifest SHA-256: 85b274a7a62a7d05b4ac5391dd4d949f446da25789fba5e1b666f2f8fb5e0ae3.
- The examples, manifest, generated responses, metrics, weights, and logs are in the local .lumi-data/experiments/conversation-response-smoke-v1 directory, outside the repository and every worktree. No external dataset, model generator, or pretrained asset was used.
- Exact wording in the goal is explicitly illustrative. It must not be treated as the fixed wording of Lumi's product responses.

## Candidate and training

- Model: single-layer tanh recurrent byte-level encoder-decoder with a shared 24-dimensional input embedding, 32-dimensional hidden state, and autoregressive output over 256 UTF-8 byte values plus EOS.
- Initialization: seeded NumPy Gaussian random weights; no pretrained weights, tokenizer, embeddings, checkpoints, or GPU.
- Size: 18,321 parameters; saved compressed artifact size 69,972 bytes.
- Recipe: seed 1729, Adam, learning rate 0.01, 300 epochs, 1,200 per-example updates.
- Training objective: mean byte-level cross-entropy fell from 5.5718046 to 0.00409686.
- Training time: 3.36 seconds in the recorded process.

## Results

The model saved and reloaded successfully. Both development prompts produced nonempty text, terminated with EOS, and decoded as valid UTF-8. These are format and execution checks, not response-quality scores. The model reused the weather reply for the held-out media comment. For the ambiguous playback prompt, it returned the training example's generic uncertainty wording instead of the more useful question about which item to play. This is a clear response-selection failure on the two illustrative cases; no exact-string match metric or human quality review was used.

| Measure | Run 2 |
|---|---:|
| Parameters | 18,321 |
| Saved weights | 69,972 bytes |
| Artifact SHA-256 | a0d53170c9f4c34e383d88b9fc1fc8cc431327306acfc07aeab2fb58b315fe77 |
| Parameter-state SHA-256 | 884bad1779c62f9000b81e3393061d9a1c8c719d7b975e6f675dcc5af6eac0e7 |
| Reload time | 12.92 ms |
| First post-reload generation | 1.64 ms |
| Warm generation latency, 40 calls | p50 0.72 ms; p95 1.31 ms |
| Process working set before / after reload | 36,401,152 / 36,495,360 bytes |
| Peak process working set | 37,572,608 bytes |

The working set includes Python and NumPy; it is not model-only memory. Timings were recorded after training in one process, so cold process startup, idle use, unload/wake, long prompts, other CPU classes, and GPU performance are unmeasured. The CPU model identifier was unavailable in the sandbox. A second same-seed run produced the same saved artifact bytes and parameter state.

## Interpretation and next decision

This closes the engineering question that a randomly initialized model can produce and reload response text on CPU. It does not establish conversational understanding, appropriate clarification, unsupported-request handling, Japanese, code-switching, multi-turn reference resolution, structured output, grounding, or ZenStream integration. The development outputs directly show that falling training loss and valid UTF-8 are not sufficient for response quality.

Do not scale this candidate or present its two examples as a benchmark. The next quality-bearing comparison needs permitted training material and a frozen, semantically diverse development set with two independent semantic reviews and separate qualified Japanese/bilingual review. The existing 101-case assistant-generated draft came through Codex using the owner's personal ChatGPT Plus account, but the exact serving model/build is unknown; the owner has directed that those drafts and all derivatives remain excluded from training, tokenizer fitting, evaluation, filtering, and release evidence. Human reviewer availability does not change that exclusion.

To reproduce, use the Python 3.12 environment with NumPy 2.3.5, keep data and output paths outside all Git worktrees, and run the random-init conversation response smoke script with the local examples and manifest. The exact input hashes, parameters, loss history, generated outputs, and artifacts are preserved in the local run directories. A follow-up varied the initialization seed across three additional runs; see [the seed-sweep report](CONVERSATION_RESPONSE_SEED_SWEEP.md).
