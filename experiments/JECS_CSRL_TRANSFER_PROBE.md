# JECS to Japanese-English retrieval-query transfer probe

**Run date:** 2026-10-03
**Evidence level:** Bounded external transfer diagnostic
**Result files:** Raw query text, item-level records, and detailed metrics stay under local .lumi-data; only this aggregate report and provenance metadata are committed.

The detailed local JSON report has SHA-256 `3c2386241b1b2c1ac3a0539aeae7fa1148737e127b50a68d3e42e6935a4041ff`.

## Result

On the same 158 Japanese-English HumanEval public-test queries, the existing JECS bilingual baseline scored **5.9185 bits per UTF-8 byte** and the existing code-switch-augmented checkpoint scored **5.0158 bits per UTF-8 byte**. The augmented checkpoint's aggregate difference was **−0.9027 bits per byte**. Lower is better.

Across paired per-query differences (augmented minus baseline), the mean was −1.3163 bits per byte, the median was −0.6759, and a seeded 3,000-resample percentile bootstrap interval for the mean was [−1.5804, −1.0641]. This interval describes variation across these 158 queries; it does not establish performance on other tasks or populations.

The change is a useful transfer signal: JECS code-switch exposure improved next-byte prediction on this separately authored Japanese-English technical-query slice. It is not evidence that Lumi can answer naturally, understand ZenStream actions, or handle spontaneous code-switching in conversation.

## Data and provenance

The only CSR-L content used was queries_ja_en/test-00000-of-00001.parquet from the pinned [Hugging Face dataset revision](https://huggingface.co/datasets/UTokyo-Yokoya-Lab/HumanEvalRetrieval-CSR-L/tree/d7634f8e08cf9249cdb2169fb153b0de85d705c8). The file is 26,901 bytes with SHA-256 e1dc7284c97c0d4e6dfd7eae7ab2dd1f5af223523b9dad23f0b41c40e2f6aba0; its normalized 158-row JSONL is also pinned locally by SHA-256 e1e5d2098393141c63b77d3019b6c2b47ccd60798b293ae36b0b339cdef1060c.

The pinned dataset README declares MIT, identifies the Japanese-English query configuration, and attributes its HumanEval source to an MIT release. The [ACL paper](https://aclanthology.org/2026.findings-acl.636/) describes the code-switch queries as author-written rewrites with a second bilingual annotator validating, editing, or discarding them. Its Japanese-English HumanEval test slice has 158 queries. The committed collection-level manifest has SHA-256 `1c4c5f6cca1e71795f879752ed60a8062548fe2d9a31d4fb10a82201962de7c4`; local per-item records have SHA-256 `681ec6a4068532f1e832972f0bcd469e480a97949eaf3dab8fe37e372287b906`. This admission is limited to local evaluation. It does not authorize training, tokenizer fitting, filtering, generation, tuning, use as a sealed holdout, or public distribution of model artifacts.

The provenance validator accepted all **158 item records**. They have 158 unique IDs and text hashes, zero exact UTF-8 hash overlap with the 2,977 eligible JECS samples, and no email or URL pattern matches. A generic phone-like-number scan flagged five strings across two rows; inspection of those exact rows found only code/date examples, not contact information. Their immutable text hashes are pinned in the local item records and the evaluator refuses changed matches.

The existing **101-case Codex draft and all derivatives were not read or used**. No raw query text, predictions, or weights were added to Git.

## Method

The probe loaded the two already-trained 18,086-parameter JECS checkpoints. Both used random initialization, seed 314159, eight epochs, 608 optimizer updates, and 19,288 training sample presentations. The baseline had no code-switch training rows; the augmented checkpoint used 443 JECS code-switch rows. This probe performed no training or tuning.

Each exact query was UTF-8 encoded and scored with the cs language tag, BOS, and EOS. The metric is teacher-forced next-byte negative log likelihood, normalized by the number of UTF-8 query bytes. No generated response, retrieval corpus, relevance label, or answer was read. The evaluator creates item-level provenance records and runs provenance/validate.py before it loads the checkpoints.

On this Windows 11 CPU host, aggregate scoring of 43,753 query bytes took 0.368 seconds for the baseline (about 118,761 bytes/second) and 0.295 seconds for the augmented checkpoint (about 148,472 bytes/second). The complete probe, including paired per-query scoring and bootstrap calculation, took 3.43 seconds. These are small batch-scoring measurements, not assistant-generation latency or a production CPU capacity result.

## Limits and next evidence

The comparison is one seed and one small public retrieval-query task. The JECS mixture experiment did not control per-language exposure, and these technical search queries are not media requests or dialogue. Public test data is not a sealed holdout. A human-reviewed, task-matched EN/JA/code-switch set with real ZenStream requests is still needed for response quality, intent/slot accuracy, clarification, unsupported requests, false actions, and grounding.

The result therefore supports only this narrow claim: **on this pinned public query slice, the existing code-switch-augmented JECS checkpoint had lower UTF-8 next-byte loss than its bilingual baseline**. It does not change the public-weight release gate; full data lineage, license, artifact, and distribution review remains required.
