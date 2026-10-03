# Random-init conversation response seed sweep

**Date:** 2026-10-03
**Evidence level:** 1 — exploratory
**Decision:** reject this tiny candidate for product use; do not scale it

## Question

Were the response-selection failures in the initial smoke peculiar to one random initialization, or did they persist across several random seeds?

## Controlled setup

- Model and implementation: the byte-level tanh encoder-decoder from [the original smoke](CONVERSATION_RESPONSE_SMOKE.md), source revision `2d4ec6f39ed6cf77186b7a71f0e277e35b25e21f`.
- Data: the same four user-provided illustrative training examples and two family-disjoint illustrative development examples. Dataset SHA-256 `974d45ea0a2b650a5246fbbaeea5f2d4edf0c1db933a511041456218517acaaa`; source manifest SHA-256 `85b274a7a62a7d05b4ac5391dd4d949f446da25789fba5e1b666f2f8fb5e0ae3`.
- Recipe held fixed: CPU, NumPy 2.3.5, Python 3.12.14, 18,321 parameters, Adam, learning rate 0.01, 300 epochs, 1,200 updates, 40 latency repeats.
- Seeds: 11, 29, 47, and the original 1729 run. Each seed changes initialization and the deterministic training-example order; no other setting changed.
- No external or pretrained material was used. The examples and run outputs remain outside Git in `.lumi-data/experiments/conversation-response-smoke-v1/`.
- No final holdout was accessed. No exact-string metric or human quality review was used.

## Results

| Seed | Final byte cross-entropy | Weight bytes | Weight SHA-256 | Warm p50 / p95 (ms) | Peak process working set (bytes) |
|---:|---:|---:|---|---:|---:|
| 11 | 0.006556804 | 69,931 | `b46e5a2899d1fb42e497fd5b2671a424f989fcc4831203e4885a6f8a75653aa0` | 0.662 / 0.819 | 37,658,624 |
| 29 | 0.007743543 | 70,024 | `333149df96d55d6efc99e3fe829c7150389c0e49438e6fdb74a71c19460fbd63` | 0.743 / 1.659 | 37,588,992 |
| 47 | 0.002994163 | 70,047 | `248adf4e57a07a84cfaf5556525bcf2855c3604238d7202d42e8e06629e970a1` | 0.848 / 1.685 | 37,707,776 |
| 1729 | 0.004096862 | 69,972 | `a0d53170c9f4c34e383d88b9fc1fc8cc431327306acfc07aeab2fb58b315fe77` | 0.649 / 0.780 | 37,642,240 |

All four runs saved, reloaded, and emitted EOS-terminated valid UTF-8 for both development inputs. That verifies execution and format only. Inspection of raw outputs found that no seed gave an on-target reply for either case: the non-action episode comment received an unrelated greeting/weather response; the ambiguous playback request received generic uncertainty in three runs and an unsupported-capability refusal in one. Seed 29 also generated the malformed greeting `Yeam and your media.`

## Interpretation

Low byte loss was consistent across seeds but did not produce the intended conversational behavior on these two examples. The seed variation confirms the failure is not just the original initialization; it does not separate architecture limits from the severe training-data coverage limit. Two illustrative cases are not a benchmark and do not support a numerical quality claim.

Do not add parameters or training epochs based on this result. The next quality-bearing comparison needs separately permitted training material and a frozen development inventory reviewed by two independent semantic reviewers and a distinct qualified Japanese/bilingual reviewer. Small level-1 engineering experiments may continue without that inventory, but must remain clearly exploratory.

## Reproduction artifacts

The exact reports, training curves, generated JSONL, model metadata, and `.npz` weights are preserved outside the repository under:

`C:\Users\mrhom\Documents\VSCode\zenstream\.lumi-data\experiments\conversation-response-smoke-v1\seed-11`, `seed-29`, `seed-47`, and the existing `run-3` directory for seed 1729.
