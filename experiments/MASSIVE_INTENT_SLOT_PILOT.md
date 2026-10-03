# MASSIVE English/Japanese intent and slot pilot

**Date:** 2026-10-03

**Evidence level:** 1 — exploratory
**Decision:** Reject this model as a Lumi candidate; retain the run as negative diagnostic evidence.

## Scope

This run asks whether the existing tiny random-initialized byte-level recurrent encoder-decoder can learn a narrow structured intent/slot target from aligned English and Japanese assistant utterances. It does not generate conversational replies. The source is the pinned MASSIVE 1.1 archive audited in [MASSIVE_AUDIT.md](MASSIVE_AUDIT.md), licensed CC BY 4.0 with the recorded attribution. Only the deterministic 256 official-train ID families and 64 disjoint public-dev ID families were used, represented in both `en-US` and `ja-JP`. The public dev slice is an exploratory diagnostic, not a Lumi-qualified development set or final holdout.

The input consists of locale plus utterance text. The target is compact JSON containing the source intent and slot names/values parsed from the annotated utterance. No upstream checkpoint, pretrained embedding, tokenizer, public test row, or excluded 101-case Codex draft or derivative was used. The exact serving model/build for that draft is unknown; the owner has directed that the draft remain excluded from training, tokenizer fitting, evaluation, filtering, and release evidence.

## Provenance and run configuration

The runner rechecks the pinned archive and member hashes, builds the selected sample records, and invokes `provenance/validate.py` before training. Validation passed with 640 item records: 512 train rows from 256 paired families and 128 dev rows from 64 disjoint paired families. The raw archive, transformed rows, item records, predictions, and weights remain in workspace-level controlled storage outside Git.

| Setting | Value |
|---|---|
| Initialization | Random; seed `314159` |
| Model | Byte-level one-layer tanh RNN encoder-decoder; 24-dimensional embeddings, 32 hidden units |
| Parameters | 18,321 |
| Objective | Per-example mean next-byte cross-entropy |
| Optimizer | Adam, learning rate `0.01`, global gradient norm clip `1.0` |
| Training | 20 epochs; 10,240 updates; one CPU thread |
| Training loss | `5.5509` initial, `0.6359` final |
| CPU generation latency | `0.790 ms` warm p50, `1.184 ms` warm p95 over 128 rows |
| Peak process working set | `54,611,968` bytes, including Python and NumPy |

This is one seed on a small sample. The latency is a local Python/NumPy microbenchmark and does not represent an integrated service or standalone model memory.

## Development results

| Measure | Result |
|---|---:|
| Required output shape | 128/128 |
| Valid UTF-8 and EOS | 128/128 |
| Valid JSON | 128/128 |
| Exact intent | 0/128 |
| Exact slot object | 42/128 |
| Slot-value micro F1 | 0.000 (`TP=0`, `FP=0`, `FN=130`) |
| Exact intent by locale | `en-US` 0/64; `ja-JP` 0/64 |
| Slot-value micro F1 by locale | `en-US` 0.000; `ja-JP` 0.000 |

The model learned to emit parseable structured output but failed to recover any correct slot value and did not predict a correct intent. Lower training loss and valid serialization therefore do not indicate useful cross-example generalization. This result does not justify a hyperparameter sweep or architecture selection on this formulation.

## Reproducibility record

- Archive SHA-256: `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`
- English member SHA-256: `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e`
- Japanese member SHA-256: `c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79`
- Source manifest SHA-256: `2e2901bbcf479b0059b18b799828a9a24612d79a853156e008ccebe4a9d1bd52`
- Selected item-record JSONL SHA-256: `d3695d3d239d19fa86e353293dbd52e3626f69622834f54c4555dbe257d2f029`
- Transform config SHA-256: `2e507e0ae639c3a5d552bfde21d93066ab36ff4d715c068ff69a3e598dc429c2`
- Selection manifest SHA-256: `7c1a7a28dcac0312e7492d138d9ca69d120baa60df0b4ab63fcf7f2db489b7f0`
- Pilot script SHA-256: `8b4ab84cb1f2d97bf4e93e700f10f9bfdba93d32bf23b5bd8506e9eeb8cbf3a7`
- Shared engine SHA-256: `5e5adf003045674a4cdf36d866c4e42613c0efdd7067298a66cce034d77d82ee`
- Local weight SHA-256: `46d82e3008005624c4066392b950b8958838049c74acb6674ab5f8e609facac3`
- Local prediction JSONL SHA-256: `dc0a8ebe0db0e489160d81e2a05d800433f3c41d500077b99d0b02255ec56aae`

The first complete local report and model output live under `.lumi-data/experiments/massive-intent-slot-pilot-v2/`; a committed-tree rerun is under `massive-intent-slot-pilot-v3/`. Neither is committed. The rerun reproduced the source, manifest, record, transform, selection, code, weight, and prediction hashes, as well as the training curve and every development metric. Training took 46.42 seconds and 46.28 seconds across the two runs. Warm p50/p95 generation timing varied from `0.790/1.184 ms` to `0.955/1.293 ms`; these remain local microbenchmarks. This confirms repeatability for one fixed seed and host, not robustness across seeds or deployment environments.

## Limits and next step

MASSIVE consists of English virtual-assistant utterances and professional localizations. It is not spontaneous Japanese dialogue, meaningful Japanese-English code-switch conversation, ZenStream media dialogue, or a source of trusted catalog state. These results say nothing about natural responses, false actions, clarification, unsupported requests, authorization, or grounding. No qualified Japanese naturalness review was performed. The public CC BY 4.0 source grant does not settle whether this trained weight artifact reproduces protected expression or whether the weights may be publicly distributed; the weights stay local pending a separate artifact-lineage and distribution review.

Use this result to reject the tiny recurrent structured-prediction candidate and prioritize a Lumi-specific, independently reviewed task dataset. Do not tune on or promote the public MASSIVE dev slice into the final holdout.
