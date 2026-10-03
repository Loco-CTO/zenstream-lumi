# JMultiWOZ scaled random-initialization pilot

**Evidence level:** Exploratory. This follow-up checks whether scaling the existing tiny random-init Japanese state-and-response candidate from 64 to 512 training dialogues changes a few public-dev diagnostics. It is not a Lumi quality evaluation, a matched model comparison, or an architecture selection.

## Data boundary and selection

The source is the pinned JMultiWOZ v1.0 release documented in the [original pilot](JMULTIWOZ_PILOT.md) and [source manifest](../provenance/data_sources.json). This run selected 512 dialogue IDs from the official training split and 64 from the official development split. The 64 development IDs were deterministically selected by SHA-256 rank after excluding the 24 public-dev dialogue IDs used in the original pilot. The run's selection manifest records all IDs and hashes; a separate integrity check found no train/dev ID overlap and no overlap with the prior dev sample. The official test split was not used.

Each dialogue contributes its final annotated SYSTEM turn: up to four preceding USER/SYSTEM utterances form the context, and a compact JSON target contains the final active domain, non-null active-domain arguments plus general city, and the Japanese SYSTEM response. Phone-like spans were replaced with `[PHONE]`; phone, address, and reference slots were omitted. No tokenizer was fit. The model consumes UTF-8 bytes directly.

The prior 101-case Codex draft and all derivatives were excluded from data selection, training, tokenizer fitting, evaluation, filtering, prompts, and release evidence. The trained weights, source archive, selection manifest, transformed records, and predictions remain in controlled local storage outside Git. Public artifact distribution remains unresolved pending a fresh rights, provenance, artifact-lineage, and release review.

## Run configuration and provenance

The scale scope was fixed to seed `314159`, 20 epochs, Adam learning rate `0.01`, greedy decoding with a 512-byte cap, and a ceiling of 10,240 per-example Adam updates. It used one CPU thread, no GPU, and Gaussian random initialization. The model is the existing single-layer tanh recurrent byte-level encoder-decoder with 18,321 trainable parameters. This run did not sweep seeds or hyperparameters.

| Evidence item | Value |
|---|---|
| Upstream revision | `a78dd334b907e12318b76bb23d0a9ed1498b8d15` |
| Archive SHA-256 | `283dea36f5abeea8f016fea2bd4df4dfba67a40f89c4600e47ed701c57ea2bc2` |
| Source-manifest SHA-256 | `fcf0b686a3fc40136132ae022e463a1a4112bfaf156076ccbf60f42930fdc222` |
| Sample-record file SHA-256 | `6def6586d6e1401481725b02731f44b5f9e84070a2d6373981cc0dafc3e08e2a` |
| Selection-manifest SHA-256 | `22d04ecf51d613a17267c052abe975971d46bb4bca6b5c79dfa0171376ce3843` |
| Transformation-config SHA-256 | `0cb2ddd1a463901d89c640ad2cee4f8479787da3b3613a07672824e6c5da1c29` |
| Pilot-script SHA-256 | `a2ba03db72ea267fc7db46c625a87d2bf896bce41bf7bec7ffccb1491160983d` |
| Shared response-engine SHA-256 | `5e5adf003045674a4cdf36d866c4e42613c0efdd7067298a66cce034d77d82ee` |
| Provenance validation | valid; 576 transformed sample records |

The local provenance validator checked source IDs, split/use declarations, source hashes, and transformed sample records before training. It does not establish Japanese response quality or settle future artifact distribution. Environment: Windows 11, AMD64, AMD Ryzen processor identifier `AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD`, 16 logical CPUs, Python 3.12.14, NumPy 2.3.5.

Reproduction requires the same pinned archive and a fresh controlled output directory outside every Git worktree:

```powershell
python experiments/random_init_jmultiwoz_pilot.py `
  --archive <controlled-path-to-JMultiWOZ_1.0.zip> `
  --output-dir <new-controlled-output-directory> `
  --train-dialogues 512 `
  --dev-dialogues 64 `
  --exclude-prior-dev
```

The scale scope rejects different train/dev counts, a missing prior-dev exclusion, or changes to seed, epochs, learning rate, and output-byte cap.

## Results

| Measure | Result |
|---|---:|
| Training / fresh development dialogues | 512 / 64 |
| Prior public-dev dialogues excluded | 24 / 24 |
| Train/dev family overlap | No |
| Training / development input-and-target UTF-8 bytes | 305,851 / 37,015 |
| Phone-like spans redacted | 141 |
| Training objective | 5.5573 initial → 0.5729 final |
| Updates / parameters | 10,240 / 18,321 |
| Valid UTF-8 / EOS / JSON / required output shape | 64 / 64 / 64 / 64 |
| Exact domain match | 17/64 (26.6%) |
| Exact argument-object match | 0/64 |
| Human-reviewed Japanese response quality | No |
| Training time | 147.85 s |
| Reload / warm inference latency | 13.08 ms / p50 2.34 ms / p95 3.28 ms |
| Process peak working set | 82,735,104 bytes |
| Saved model | 70,586 bytes; SHA-256 `a7b88e014a6b2df179df43bc553d1f54a1c6437311a03d90ce0aebd0943f7fe8` |
| Development prediction JSONL SHA-256 | `3af8b3a39e8f4d465aaa39c3f76ae9946431d9a7dab499ed3343f6c57eae12cd` |

The output-shape result is an engineering diagnostic. Exact argument matching remained zero despite an eightfold increase in training examples and updates over the first run. Exact domain accuracy was 26.6%, which is not evidence of broad task generalization; Japanese response meaning and naturalness were not independently rated. This candidate remains rejected as evidence of useful Lumi behavior. The 64-item public sample is not the qualified, independently reviewed development set.

Compared with the earlier 64/24 run, this run changes both training size and the public-dev sample. The raw scores are descriptive only and cannot isolate a causal benefit from adding training data. Both Japanese pilots remain narrow travel-domain diagnostics; neither tests Lumi media actions, English, natural EN/JA code-switching, no-action behavior, false-action risk, clarification, unsupported requests, authoritative ZenStream grounding, or production lifecycle costs.

The model file, metadata, training curve, selection manifest, provenance records, validator output, report JSON, and raw predictions are retained outside Git under the controlled scale-pilot output directory. Do not publish these weights based on this pilot; complete the intended public-distribution review against the exact artifact and full lineage first.
