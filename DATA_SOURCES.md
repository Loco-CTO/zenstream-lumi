# Lumi data sources and provenance

**Current data status:** One exact ASDC Japanese dialog is approved for a local public development diagnostic; its text and item-level provenance remain in controlled storage. No external content has been used for training or tokenizer development. MASSIVE 1.1 is pinned and audited as a candidate only; no MASSIVE text has been admitted. The separate 101-case synthetic evaluation draft remains in local controlled storage with unresolved rights, no independent review, and no scoring eligibility. Its metadata-only fingerprint is recorded in [the draft inventory manifest](evaluation/draft_inventory_manifest.json). The public [data-source manifest](provenance/data_sources.json) records the ASDC source; synthetic generations have a separate [manifest](provenance/synthetic_data.json) and [schema](provenance/synthetic_data.schema.json). Research references are kept separately in [RESEARCH_REFERENCES.md](RESEARCH_REFERENCES.md) and are not training data.

## Admission policy

Do not use an external dataset, benchmark, corpus, tokenizer-training corpus, metadata set, filtering set, or other content until its origin, exact version, license, attribution, and permitted uses have been reviewed and recorded. Public accessibility alone is not evidence of permission. Exclude sources whose provenance or license cannot be determined with reasonable confidence.

For each source, record its canonical URL/identifier, authors or organization, version/revision, release and access dates, license and license text URL, copyright/attribution terms, commercial/redistribution/modification/derivative permissions, and separate `training_use` and `evaluation_use` permissions. Also record Lumi uses, transformations, filters, deduplication, approximate documents/examples and token contribution, and direct/indirect inclusion. Preserve a checksum for the exact acquired source or derived shard when practical.

Separate and label each source's use as pretraining, instruction training, ZenStream-specific training, tokenizer training, evaluation, filtering, synthetic generation, or another declared purpose. Run `provenance/validate.py` before scoring or assembling an approved dataset; it rejects unresolved source/generator IDs, split mismatches, unapproved use permissions, and changed evaluation-case content hashes. Every production training run must also point to a frozen manifest revision and the processing pipeline revisions used to produce its exact inputs.

## Synthetic data

The local draft bundle records one interactive assistant-generation event and per-case sample hashes, while leaving permissions unknown and all rows unapproved. No synthetic content has been admitted for training, tokenizer development, or scoring. Before any use, record the provider, model and dated version, generation date, purpose, prompt family/method, important settings, number of examples, filtering, deduplication, automatic validation, manual review, downstream split/use, and relevant provider terms in [the synthetic-data manifest](provenance/synthetic_data.json). Treat generated text as untrusted: validate semantics, rights, privacy, diversity, contamination, and label correctness before inclusion.

For any new human-written material, use the [human contributor intake checklist](provenance/AUTHOR_CONTRIBUTION_CHECKLIST.md) to settle use scope and artifact-distribution disclosures before collection. An internal evaluation-only grant is distinct from permission to train or distribute model weights; do not widen one into the other.

## Evaluation and contamination

Evaluation sources are provenance sources too. Keep development and final-holdout access and use explicit; neither holdout examples nor their answers may enter training, generation prompts, filtering, or tuning. Preserve source and license records even for data that is used only to score a model.

## Release summary

There is one approved development-evaluation item: ASDC main dialog 001, as recorded in `provenance/data_sources.json` and its controlled item-level record. There are no approved training or tokenizer sources. MASSIVE is not yet admitted, and the controlled 101-case draft inventory is not an approved evaluation source or part of a release. Update this statement for each release from reviewed manifests, not from undocumented developer knowledge.
