# Lumi provenance records

The public repository contains provenance formats and research documentation, not training text. `data_sources.json` is the collection-level catalog of external, authored, or synthetic sources. It is intentionally empty until a source passes review.

`document_records.schema.json` defines one JSON object per released, processed, or authored sample. Store records as newline-delimited JSON in a controlled data-build artifact, with one line per exact sample made available to training, tokenizer learning, generation, or evaluation. Do not add source text to the public repository. For each record, retain stable source item IDs, source and revision URLs where available, hashes of source and resulting text, rights evidence, permitted uses, attribution, transformations, split, privacy review, and upstream record IDs. A mixed document assembled from multiple inputs must list every parent source record.

Each training run must pin the source manifest, document-record artifact, transformation code/configuration, tokenizer inputs, and split manifests by revision and SHA-256. Keep development and final holdout records outside training, filtering, synthetic generation, and tokenizer input. When rights or privacy terms prohibit exposing an item-level ledger, retain the full ledger in controlled storage and publish its immutable hash plus a safe summary.

No sample-level records exist yet because no data has been approved or processed.
