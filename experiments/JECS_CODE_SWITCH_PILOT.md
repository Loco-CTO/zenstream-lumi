# JECS text-only code-switch byte-LM pilot

**Status:** Completed local exploratory ablation, 2026-10-03. This experiment measures next-byte fit on one small, acted/read corpus. It is not evidence of natural conversation, ZenStream task performance, product readiness, or model-release clearance.

## Question and comparison

Does including the JECS code-switch text change held-out UTF-8 byte loss for a tiny random-initialized byte language model, compared with training on its aligned Japanese and English rows only?

Both candidates use the same one-layer tanh recurrent language model, 18,086 parameters, random initialization, seed 314159, eight epochs, batch size 32, Adam at 0.003, and global gradient clipping at 1.0. Each receives 19,288 sample presentations and performs 608 optimizer updates. The bilingual baseline resamples Japanese and English training rows with replacement to meet the fixed presentation budget. The augmented run sees all training rows once each per epoch, including the selected code-switch rows. This is a data-mixture comparison: Japanese and English example exposure differs between candidates, so it does not isolate a pure code-switching effect while holding every language's exposure constant.

The model reads UTF-8 bytes with fixed BOS, EOS, and language-tag IDs. It learns no tokenizer and uses no pretrained weights, embeddings, external model, or response-generation data. All training and scoring ran on CPU.

## Data and split

The admitted source is [JECS v1](https://sites.google.com/site/shinnnosuketakamichi/research-topics/jecs_corpus), limited to the three hashed neutral transcript text members and README in [`provenance/data_sources.json`](../provenance/data_sources.json). The author page licenses transcript text under [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) and treats speech audio separately. The text is based on JEC Basic Sentence Data; the [JEC resource page](https://nlp.ist.i.kyoto-u.ac.jp/EN/?JEC+Basic+Sentence+Data=) and [JAIST language-resource catalog](https://www.jaist.ac.jp/project/NLP_Portal/doc/LR/lr-cat-e.html) provide its attribution context. Only the exact neutral text files and README were acquired. Audio and other archive members were not used.

There are 1,219 Japanese rows and 1,219 aligned English rows across 1,219 numeric utterance families. The 539 code-switch rows belong to those families. The deterministic split hashes `jecs-v1|314159|<family_id>` and assigns a family to development when the first four digest bytes, interpreted as a big-endian integer, are divisible by five. This gives 984 training families and 235 development families; all language rows from a family stay together.

| Language rows | Training | Development |
|---|---:|---:|
| Japanese | 984 | 235 |
| English | 984 | 235 |
| Code-switch | 443 | 96 |

The row parser verified source counts and aligned IDs, rejected duplicate normalized text within a language, and found zero identical normalized sample hashes crossing train and development or appearing across language files. The email, URL, and phone-like sequence scans found no pattern hits. These checks do not prove the absence of personal information or semantic duplication; the exact family split only checks leakage inside this JECS cohort.

## Results

The metric is mean negative log likelihood in bits per UTF-8 byte, calculated only over text-byte targets. Lower is better for this byte-prediction task. It is not a response-quality score.

| Development rows | UTF-8 bytes | JA+EN baseline | JA+EN+CS |
|---|---:|---:|---:|
| Japanese (235) | 13,998 | 2.7661 | 2.7422 |
| English (235) | 11,858 | 3.4410 | 3.4412 |
| Code-switch (96) | 5,361 | 5.4271 | 3.1414 |

The augmented candidate has much lower byte loss on the held-out JECS code-switch rows, while English is essentially unchanged and Japanese moves slightly. The baseline never saw code-switch training rows, so its higher code-switch loss is expected and is not evidence of generalization to natural code-switching. There is one seed, a small public development subset, and a fixed simple model; the result is a pilot observation, not a reliable estimate of model quality.

| Candidate | Training seconds | Parameters | Compressed weights | SHA-256 |
|---|---:|---:|---:|---|
| JA+EN baseline | 9.81 | 18,086 | 68,842 bytes | `12c2213ba4d720da560bd920fc1ef2fa264605542872fa48f5ceff76e77e7ca6` |
| JA+EN+CS | 9.51 | 18,086 | 68,769 bytes | `fa0dc5cf67380ad6528d373f0ab374c3470628908c0910d2eda2172b85a06f4c` |

Runtime: Python 3.12.14, NumPy 2.3.5, Windows 11, CPU only, 16 logical CPUs reported. Process working set was 45,961,216 bytes before training and 53,432,320 bytes after; process high-water working set was 68,497,408 bytes both before and after. These process-wide readings include Python and NumPy and cannot be attributed to the model alone. Training took about ten seconds per candidate on this machine; it does not predict full-model training or inference requirements.

## Provenance and reproducibility

Raw transcript text, weights, predictions, per-sample provenance, and split selections remain in controlled local storage outside Git under `.lumi-data/jecs-v1/pilot-20261003/`. The committed experiment code verifies the four exact source-file hashes before processing and refuses output paths inside a Git worktree. It writes hashes and metrics, not utterance text.

| Artifact | SHA-256 |
|---|---|
| Pinned source manifest snapshot | `17f3e32d5804088ce970b118d91f01916a45863cbc03527be3065206800a5eb0` |
| Local `document_records.jsonl` | `75845f421d19ed429abd6b99c5b229d315057939d1c2332e19ce81ad951b991e` |
| Local `selection_manifest.json` | `f6261441684eb43bda0ad67b79d08dd4a296b6058c02f727c2e9574afb7f75e3` |

The standard provenance validator accepted all 2,977 item records: 4 sources, 0 synthetic generations, 2,977 sample records, and no evaluation cases. The full run report, including per-language counts, byte-level losses, CPU timing, and local artifact hashes, is `report.json` in the controlled output directory.

Re-run after placing only the pinned four text/evidence files in a controlled directory:

```powershell
python experiments/jecs_byte_lm_pilot.py --data-dir <controlled-jecs-text-dir> --output-dir <outside-git-output-dir>
python provenance/validate.py --records <outside-git-output-dir>/document_records.jsonl
```

## Limits and next research step

JECS consists of acted/read parallel utterances with contributor- and speaker-designed code-switch segments, not spontaneous conversation, current colloquial Japanese, or ZenStream media requests. The score uses only public development families, not a sealed holdout. The study does not evaluate instruction following, response naturalness, grounding, unsupported requests, safety, privacy leakage, or catalog actions. It neither fits a tokenizer nor generates responses. No trained weight is a candidate for distribution: review complete data lineage, source-specific obligations, memorization and extraction risk, and the exact artifact before any public release.

Next useful evidence is a multi-seed comparison plus independently reviewed natural, task-matched English/Japanese/code-switch dialogue that is separately authorized for the intended training and public artifact use. Keep this source diagnostic separate from that evaluation.
