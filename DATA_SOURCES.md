# Lumi data sources and provenance

**Current training-data status:** No external dataset content has been admitted or used for training, tokenizer development, or evaluation. The machine-readable [data-source manifest](provenance/data_sources.json) has an empty source list and is described by [its JSON Schema](provenance/data_sources.schema.json). Synthetic generations have a separate [manifest](provenance/synthetic_data.json) and [schema](provenance/synthetic_data.schema.json). Research references are kept separately in [RESEARCH_REFERENCES.md](RESEARCH_REFERENCES.md) and are not training data.

## Admission policy

Do not use an external dataset, benchmark, corpus, tokenizer-training corpus, metadata set, filtering set, or other content until its origin, exact version, license, attribution, and permitted uses have been reviewed and recorded. Public accessibility alone is not evidence of permission. Exclude sources whose provenance or license cannot be determined with reasonable confidence.

For each source, record its canonical URL/identifier, authors or organization, version/revision, release and access dates, license and license text URL, copyright/attribution terms, commercial/redistribution/modification/derivative permissions, Lumi uses, transformations, filters, deduplication, approximate documents/examples and token contribution, and direct/indirect inclusion. Preserve a checksum for the exact acquired source or derived shard when practical.

Separate and label each source's use as pretraining, instruction training, ZenStream-specific training, tokenizer training, evaluation, filtering, synthetic generation, or another declared purpose. Every production training run must point to a frozen manifest revision and the processing pipeline revisions used to produce its exact inputs.

## Synthetic data

No synthetic training or evaluation data has been generated. Before any use, record the provider, model and dated version, generation date, purpose, prompt family/method, important settings, number of examples, filtering, deduplication, automatic validation, manual review, downstream split/use, and relevant provider terms in [the synthetic-data manifest](provenance/synthetic_data.json). Treat generated text as untrusted: validate semantics, rights, privacy, diversity, contamination, and label correctness before inclusion.

## Evaluation and contamination

Evaluation sources are provenance sources too. Keep development and final-holdout access and use explicit; neither holdout examples nor their answers may enter training, generation prompts, filtering, or tuning. Preserve source and license records even for data that is used only to score a model.

## Release summary

There are currently no training, tokenizer, or evaluation data sources to report. This statement must be updated for each release from the reviewed manifest, not from undocumented developer knowledge.
