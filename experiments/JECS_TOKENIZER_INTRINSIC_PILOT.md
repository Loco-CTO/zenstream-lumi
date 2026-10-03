# JECS tokenizer intrinsic pilot

**Date:** 2026-10-03
**Evidence level:** 1 — exploratory
**Decision:** No tokenizer selected; no Lumi behavior or quality claim.

## Scope and source

This experiment compares a direct UTF-8 byte baseline with small SentencePiece BPE and Unigram candidates. It measures intrinsic tokenization and local CPU cost only. It does not train a language model or evaluate assistant responses.

The source is JECS v1 neutral transcript text under the project’s exact local scope: the pinned Japanese, English, and code-switched transcript members and their README. The project’s source review records CC BY 3.0 attribution and adaptation obligations. This experiment used only the 984-family training split to fit vocabularies. It measured intrinsic statistics on the separate 235-family public-development split and did not fit, tune, or filter from that split. The 101-case Codex draft and all derivatives remained excluded.

The source author marks the code-switched text segment with paired asterisks; those markers were removed from tokenizer input, while their character offsets were retained for boundary measurement. JECS does not label whether each utterance is Japanese-matrix or English-matrix switching, so the report does not infer that direction from Unicode script. The corpus is acted/read text, not spontaneous conversation.

The JECS author page identifies transcript text as CC BY 3.0, and the [license deed](https://creativecommons.org/licenses/by/3.0/) permits sharing and adaptation subject to attribution and other conditions. This scope covers only this local experiment. It does not resolve rights in the resulting tokenizer/model artifacts or approve public distribution; the source and every artifact remain subject to a separate release-lineage review.

## Method

- Input: 2,411 training rows from 984 grouped families, 135,524 UTF-8 bytes including row separators. The separate development measurement used 566 rows: 235 English, 235 Japanese, and 96 code-switched rows.
- Integrity: all four pinned JECS source-file hashes matched; the parser found zero email-, URL-, or phone-like pattern hits. The provenance validator accepted 2,977 source item records, with zero evaluation cases.
- Tokenizers: direct UTF-8 bytes; SentencePiece BPE and Unigram at 2,048 and 4,096 entries. The vocabulary sizes were raised from an initial 512/1,024 plan because full character coverage required more than 1,500 training characters plus byte-fallback symbols.
- Configuration: SentencePiece 0.2.2, Apache-2.0; identity normalization, preserved whitespace, byte fallback, full character coverage, one training thread, and no pretrained components. The [tagged v0.2.2 training options](https://github.com/google/sentencepiece/blob/v0.2.2/doc/options.md) document a default seed of `4294967295`. The installed wheel rejected an explicit `random_seed` option, so the experiment records the library default and pins the package, native-module hash, input path, and model prefix. Two independent runs with the same stable paths reproduced all four model and vocabulary file hashes exactly.
- System: Python 3.12.14 on Windows 11, AMD64 Family 25 Model 97 Stepping 2, 16 logical CPUs; CPU only. Tokenizer build and load timings are local microbenchmarks, not model-training FLOPs or deployment measurements.
- Metrics: exact UTF-8 round-trip; tokens per Unicode scalar; token-count p95; English fertility using whitespace-separated words (`str.split`, 2,074 words); marked-switch boundaries inside a token; tokenizer/vocabulary and illustrative FP32 token-embedding bytes at width 32; warm row latency and tokenizer build/load time.

## Results

All candidates round-tripped every development row exactly. `Tokens/scalar` is reported in EN / JA / code-switched order. Sequence p95 is the number of tokens per row in the same order. “Boundary” is the count of source-marked boundaries lying strictly inside a token, out of the same 192 marked boundaries.

| Candidate | Tokens per scalar EN / JA / CS | Sequence p95 EN / JA / CS | EN whitespace-word fertility | Boundaries inside token | Model + vocab bytes | Build / cold load |
|---|---:|---:|---:|---:|---:|---:|
| UTF-8 bytes (256 IDs) | 1.000 / 2.920 / 2.130 | 85 / 90 / 79 | 5.717 | 0 / 192 | 0 | n/a |
| BPE 2,048 | 0.426 / 0.800 / 0.690 | 37 / 25 / 28 | 2.433 | 0 / 192 | 49,030 B | 54.2 / 15.5 ms |
| BPE 4,096 | 0.319 / 0.635 / 0.559 | 27 / 21 / 23 | 1.825 | 14 / 192 (7.3%) | 111,634 B | 39.8 / 18.0 ms |
| Unigram 2,048 | 0.455 / 0.833 / 0.752 | 41 / 26 / 33 | 2.602 | 0 / 192 | 58,047 B | 321.0 / 5.2 ms |
| Unigram 4,096 | 0.344 / 0.717 / 0.653 | 29 / 24 / 28 | 1.967 | 0 / 192 | 156,952 B | 236.0 / 10.3 ms |

BPE 4,096 used the fewest tokens per scalar on this development slice, including Japanese, but it merged 14 source-marked switch boundaries and required twice the illustrative width-32 embedding memory of the 2,048-entry models. The other subword candidates kept every marked boundary between tokens and had higher token fertility. These are corpus-specific intrinsic tradeoffs; they do not establish whether any choice improves language understanding, naturalness, task accuracy, or safety.

Warm per-row p50/p95 latency ranged from 0.0031/0.0054 ms (Unigram 2,048) to 0.0055/0.0113 ms (BPE 2,048). The byte baseline measured 0.0003/0.0008 ms. These short strings and repeated in-process calls are too small to predict user-visible inference latency; no production hardware, model decoding, memory, or idle behavior was measured.

## Artifacts and reproducibility

The raw text, sample records, reports, and fitted tokenizer files remain in `.lumi-data` outside Git. Two stable-path run reports are at `tokenizer-stable-v1-a/report.json` and `tokenizer-stable-v1-b/report.json`, under the controlled `jecs-v1` data directory. The stable model prefix and temporary training text are isolated under `lumi-jecs-tokenizer-intrinsics-v1`; the runner refuses to replace a pre-existing training-text file unless its hash matches the expected corpus. Both runs passed the provenance validator and produced identical SHA-256 values for each `.model` and `.vocab` file.

| Candidate | Model SHA-256 | Vocab SHA-256 |
|---|---|---|
| BPE 2,048 | `3fda68295ac15cf8164e60cbc231d4b89cbff7685ade58781fe72f0476145cd5` | `945ee885ae9360bce959d605db8821d4d02b3a8a14217d020b6b700b859a17fe` |
| BPE 4,096 | `1e324ddf22974b084605cdc8f451d7d134e2b3ff39a4fb32409e72f129c60324` | `75439de0f09cdcc8b8323153c0fe9b976300ab326daa6d4ab841dfbf892962cd` |
| Unigram 2,048 | `0865e3713c30b4dc8e73adbe4668676f3af95ff08cde4932fd20610217d82546` | `13751bba28dad904dd591e78d9af97e6f3f68b3a1719ac87754d0b55300c3153` |
| Unigram 4,096 | `5833054b547627b8b14e167becb432d770c4841fd21313f0ce75e9c068c43c9c` | `cb22e96527028b3a800f06fd7a1f9ccf262cdb15ffa4fbe2c16550d9af699746` |

| Lineage item | SHA-256 |
|---|---|
| Source manifest snapshot | `d9542e4f7bb4d7a06ac812e2e955bb27b35f62dcde08a5fe4c19e7f129beafd9` |
| Controlled `document_records.jsonl` | `1c2fac7387edc9f243633a7d4f2813f89b57465748c8b9f3efdf2aa2d2145b1e` |
| Controlled selection manifest | `d3b6c63260d28465da32daa8485c8d7afaecc5842cbe020282ffdf0284796137` |
| Training text | `a581559384f9ac222e1e81574beb36b0f94790ff6342af83606e6c825a778940` |
| Experiment runner | `f1fcc5cc3d805002ca720567f3a0c891d981a5a3721a78550b5b26684bce89a0` |
| SentencePiece native module | `6fb545d1295cb7a4e87ae1884ea0eb3775d8261caed4132d1db5266b113bb781` |
| Requirements file | `4e9d76444e7769ecb231c8496881195413276072c80edd3868f67a3a3156fc32` |

Pinned source members: `transcripts_ja.txt` `20551876ae990563b2fda3b107fb3c5249e485d5b01847452f26a2f7956d2b5e`; `transcripts_en.txt` `a4d2e8980d42e8747e2e670c9d112953bd8298da292ba43c1bd2ffa6f8ffc083`; `transcripts_cs.txt` `7c26eea872f3099e205035973c0784fcaa473c89f26018a5b4a1a6d72a2e468c`; `README.md` `4d68ea0502e58698e245f104098f4dd9679e07398a4d44bc95b586732536f10d`.

## Limits and next step

The source does not provide natural-dialogue labels or code-switch matrix-direction labels. This report cannot establish natural Japanese, conversational switching, English/Japanese intent or argument performance, clarification, uncertainty, unsupported-request behavior, trusted-state grounding, false-action rates, or public artifact rights. The public development split is an exploratory intrinsic diagnostic, not a Lumi-reviewed evaluation set. The final tokenizer remains unselected.

The next tokenizer comparison requires the independent, task-matched review set and a training corpus appropriate for the intended assistant. Keep this JECS result as a reproducible tokenizer engineering baseline only.
