# Lumi provenance records

The public repository contains provenance formats and research documentation, not training text. `data_sources.json` is the collection-level catalog of external, authored, or synthetic sources. It is intentionally empty until a source passes review.

`document_records.schema.json` defines one JSON object per released, processed, or authored sample. Store records as newline-delimited JSON in a controlled data-build artifact, with one line per exact sample made available to training, tokenizer learning, generation, or evaluation. Do not add source text to the public repository. For each record, retain stable source item IDs, source and revision URLs where available, hashes of source and resulting text, rights evidence, permitted uses, attribution, transformations, split, privacy review, and upstream record IDs. A mixed document assembled from multiple inputs must list every parent source record.

Each training run must pin the source manifest, document-record artifact, transformation code/configuration, tokenizer inputs, and split manifests by revision and SHA-256. Keep development and final holdout records outside training, filtering, synthetic generation, and tokenizer input. When rights or privacy terms prohibit exposing an item-level ledger, retain the full ledger in controlled storage and publish its immutable hash plus a safe summary.

No sample-level records exist yet because no data has been approved or processed.

## Provenance gate

Run the standard-library validation gate before building a dataset or scoring an evaluation set:

```powershell
python provenance/validate.py --records <document-records.jsonl>
python provenance/validate.py --records <document-records.jsonl> --cases <evaluation-cases.jsonl>
```

The gate checks unique record IDs, source and generation references, source revisions, intended-use declarations, sample-level rights and review status, synthetic prompt/model provenance, split consistency, parent lineage, and evaluation case hashes. An approved evaluation item requires `evaluation_use: permitted` in both its source record and its item-level rights record. Training inputs separately require `training_use: permitted`. Conditional or unknown permission is not enough for the automated gate to pass an approved record.

Evaluation schema v2 records map `development` cases to the `development` provenance split and `final_holdout` cases to `sealed_holdout`. Each case record's sample hash is SHA-256 over canonical UTF-8 JSON containing its schema version, case and family IDs, split, language, categories, conversation turns, trusted context (including its complete inline fixture or `null`), and gold semantic output. Case-level review references and provenance metadata are excluded from that hash. The controlled review ledger binds every decision to the same sample hash, avoiding a hash cycle while detecting edits to reviewed content. Editing any evaluation content therefore requires updating its sample-level provenance hash and repeating the affected reviews.

The gate verifies declared metadata and joins; it does not determine whether a license interpretation is legally correct, validate that an arbitrary training artifact's bytes match its declared hash, or replace native-language and subject-matter review. Keep the actual data-build artifacts and source evidence in controlled storage, and preserve their hashes with each run.
