# JMultiWOZ random-initialization training pilot

**Evidence level:** Exploratory. This pilot answers whether the existing tiny from-scratch recurrent candidate can train, save/reload, and produce schema-valid Japanese state-and-response outputs from an independently licensed dialogue source. It does not test the Lumi product contract or choose an architecture.

## Source and data boundary

The source is the authors' [JMultiWOZ repository pinned to commit `a78dd334b907e12318b76bb23d0a9ed1498b8d15`](https://github.com/nu-dialogue/jmultiwoz/tree/a78dd334b907e12318b76bb23d0a9ed1498b8d15), its [dataset README](https://github.com/nu-dialogue/jmultiwoz/blob/a78dd334b907e12318b76bb23d0a9ed1498b8d15/dataset/README.md), and the [LREC-COLING paper](https://aclanthology.org/2024.lrec-main.835/). The authors identify the dataset as CC BY-SA 4.0 and state that models trained using it are not copies or direct derivatives of the dataset. The tracked source record preserves attribution and marks modification and redistribution as conditional. The model and raw data remain local; re-review the exact intended public artifact and its provenance before any public distribution.

The local archive SHA-256 is `283dea36f5abeea8f016fea2bd4df4dfba67a40f89c4600e47ed701c57ea2bc2`. The dialogue and split member hashes are pinned in `provenance/data_sources.json` and the local selection manifest. Selection takes 64 train and 24 disjoint development dialogues by deterministic SHA-256 rank from the authors' official splits. The test split, remaining source items, backend database, account identifiers, and goal templates are not used.

Each selected item is reduced to the last four preceding USER/SYSTEM utterances and a compact JSON target containing the final annotated active domain, non-null active-domain arguments plus general city, and final SYSTEM reply. Phone-like sequences are replaced with `[PHONE]`; phone, address, and reference slots are omitted. No tokenizer is fit on this or any other corpus. No 101-case Codex draft or derivative was loaded, compared, or used for training, tokenization, evaluation, filtering, prompts, or release evidence.

Before model training begins, the runner writes one hash-only item record for each transformed sample to controlled local storage and invokes `provenance/validate.py`. The final run validated 88 records (`status: valid`; 2 source records; 88 sample records). The local document-record JSONL SHA-256 is `05e4872d8c0dae6c33962dbc18be7504b7c6557ea66a47c034e54f30f80bb382`; the frozen source-manifest SHA-256 is `13aca88db42b77cc4521ea07e3d9f7e558cd8744a4a37e3ed6d4f8d5d528cb19`. The validator confirms declared metadata and joins; it does not decide legal rights or replace human language review.

All source archives, normalized samples, predictions, item records, and model artifacts are outside Git under the controlled `.lumi-data` directory. Only manifests, transformation code, and aggregate results are committed.

## Reproduction

The locked final run used seed `314159`, 20 epochs, learning rate `0.01`, 1,280 updates, and one CPU thread. The model was initialized from Gaussian random values, with no pretrained artifacts. The implementation is a single-layer tanh recurrent byte-level encoder-decoder with 18,321 trainable parameters. Environment: Python 3.12.14, NumPy 2.3.5; GPU not used.

The local runner's source-code hashes and transformation-config hash are:

- Pilot: `5916172587b9fcededc6748874e6e48c0b3b2013fc6bbc9312c2aee0d3d604a9`
- Shared response engine: `5e5adf003045674a4cdf36d866c4e42613c0efdd7067298a66cce034d77d82ee`
- Deterministic selection manifest: `64e4ac71f382172cee4492be92512c5da3738c42017fbad8bd0468df42952d7d`
- Transformation configuration: `0cb2ddd1a463901d89c640ad2cee4f8479787da3b3613a07672824e6c5da1c29`

The training objective is the per-example mean byte-level next-token cross-entropy, averaged across examples. Adam uses β1 `0.9`, β2 `0.999`, epsilon `1e-8`, and global gradient-norm clipping at `1.0`; the learning rate is fixed for all updates. Training and generation consume UTF-8 byte IDs directly, so there is no learned tokenizer.

The pinned archive is stored outside Git. Run the script with its archive path and a fresh output directory; it fails if the archive, source, selection scope, or provenance check differs:

```powershell
python experiments/random_init_jmultiwoz_pilot.py `
  --archive <controlled-path-to-JMultiWOZ_1.0.zip> `
  --output-dir <new-controlled-output-directory>
```

The runner records its training settings, environment, code hashes, source manifest, item-record hash, selection hash, model hash, prediction hash, and provenance-validation summary in the local report.

## Results

| Measure | Result |
|---|---:|
| Training dialogues / development dialogues | 64 / 24 |
| Training / development input-and-target bytes | 40,379 / 14,397 UTF-8 bytes |
| Phone-like spans redacted | 19 |
| Training objective | 5.5576 initial → 0.4913 final |
| Updates / parameters | 1,280 / 18,321 |
| Valid UTF-8 / EOS / JSON / required output shape | 24 / 24 / 24 / 24 |
| Exact domain / argument matches | 7/24 / 0/24 |
| Human-reviewed Japanese response quality | No |
| Training time | 22.03 s |
| Warm inference latency, 24 requests | p50 2.52 ms / p95 3.17 ms |
| Process peak working set | 56,147,968 bytes |
| Saved model | 70,488 bytes; SHA-256 `fbd63613f0f30d8002a25e53c0531b305698f84c9960710bd26b0d049c52da9e` |
| Development prediction JSONL SHA-256 | `845274ccc3f591cbeb489dfa233ebf12540c03d42bb66589ed2d06a28cbd1933` |

Training loss decreased substantially, but development exact-domain accuracy was 29.2% and exact-argument accuracy was 0%. All generations were syntactically valid JSON, which is only an output-format diagnostic. No Japanese response-quality judgment was collected. These results reject this run as evidence of useful task generalization.

## Reproducibility and loader result

The locked final run ran on Windows 11, AMD64 (processor identifier `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`), with 16 logical CPUs; NumPy was constrained to one compute thread. The same seed and data pipeline produced the same parameter-state, model-artifact, and prediction hashes across the initial run and subsequent repetitions. The first loader parsed the entire 169 MB dialogue JSON member and reached a 1,071,058,944-byte process peak working set. Replacing that loader with streaming top-level JSON parsing preserved the selected records and outputs: subsequent runs measured about 55–56 MB peak working set. This is a local loader-memory comparison, not a standalone model memory measurement. The final run's wall time and small-sample latency are single-machine diagnostics, not deployment targets.

## Interpretation and next evidence

This corpus covers Japanese travel dialogues only. It does not exercise English, natural EN/JA code-switching, ZenStream media actions, no-action conversation, clarification, unsupported requests, low-confidence handling, false-action rates, trusted-state grounding, integration, cold start, unload, disabled/idle cost, or other CPU classes. Exact slot matches on 24 public development examples are too small and narrow for a product claim. No tokenizer, model formulation, or sequence architecture has been selected, and no weights are approved for public release.

The next quality-bearing comparison still needs a frozen and independently reviewed development set with two semantic reviewers and a qualified Japanese/bilingual reviewer, plus the pinned scoring and compute controls in [the architecture pilot](ARCHITECTURE_PILOT.md). Exploratory engineering work can continue without the final holdout or artifact-release review.
