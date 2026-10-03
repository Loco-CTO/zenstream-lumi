# Japanese movie dialogue source audit: JMRD and RecomMind

**Reviewed:** 2026-10-03
**Decision:** Keep both releases as unadmitted candidates for small, public Japanese development diagnostics only. RecomMind is the stronger fit for preference tracking in multi-turn conversation; JMRD is a useful lead for long, knowledge-grounded movie recommendations. Neither is a ZenStream intent benchmark or a final holdout. No corpus files were downloaded, copied, scored, or used.

This is a release-level audit of the authors' public repository pages and papers, not an item-level rights or privacy review. The repository `main` branches are mutable and have not been pinned or hashed. Do not use either release until the exact revision and selected items have been reviewed and recorded.

## Evidence and task fit

| Source | What the release and paper report | Potential diagnostic value | Limits for Lumi |
|---|---|---|---|
| [Japanese Movie Recommendation Dialogue (JMRD)](https://github.com/ku-nlp/JMRD), [2022 paper](https://aclanthology.org/2022.dialdoc-1.9/) | The paper reports 5,166 human-human dialogues, 116,874 turns, 261 movies, 322 crowdworkers, and 22.6 turns per dialogue. The current repository README instead describes about 5,000 dialogues and lists 4,575 train, 200 valid, and 300 test; it explicitly says its count and split differ from the reference papers. | A possible small public diagnostic for Japanese multi-turn coherence and knowledge-grounded recommendation language, after item selection and qualified Japanese review. | The recommender presents one selected movie in a long, detail-focused dialogue. It is a weak match for initial preference elicitation, ZenStream actions and arguments, no-action decisions, and English/Japanese code-switching. The paper says movie title/year/director/cast/plot text comes from Wikipedia, genre labels from Yahoo! Movies, and review text was collected through Yahoo! Crowdsourcing; these lineages need separate rights review. |
| [RecomMind](https://github.com/ku-nlp/RecomMind), [2024 paper](https://aclanthology.org/2024.sicon-1.4/) | The paper reports 1,201 Japanese movie recommendation dialogues, 17.5 utterances per dialogue, 739 movies, 27 recommenders, and 46 seekers. Participants annotate first-person and second-person estimates of the seeker's knowledge and interest for entities in the conversation. | The stronger of these two leads for an external Japanese conversation diagnostic focused on preference tracking and whether a response follows the seeker's stated interests. Its labels may help analyze preference-tracking errors, subject to a separate annotation and scoring plan. | It remains movie-domain dialogue, not ZenStream intent/argument extraction, playback behavior, or code-switching. The paper says movie metadata was collected from Wikipedia and Yahoo! Movies and that review material for 261 movies came from JMRD. The public README shows timestamped turns, numeric participant names described as anonymized, questionnaires, search queries, and internal-state annotations; privacy and participant-consent scope need review before selecting any item. |

The papers and current repository pages report different JMRD counts and splits. Treat the repository files as a distinct release until its exact revision is pinned and reconciled; do not transfer the paper's corpus statistics or split labels to that release. RecomMind's paper reports its collection statistics, while the public repository has no frozen revision recorded in Lumi yet.

## Rights, privacy, and contamination gates

Both author repositories declare CC-BY-SA 4.0 at repository level. That declaration is useful evidence, but it does not itself establish that every bundled field, upstream work, participant contribution, and annotation is covered for every proposed use. JMRD and RecomMind include movie descriptions and other knowledge content with separate upstream sources; RecomMind also reuses JMRD review material. The papers and repository pages reviewed here do not settle all item-level permissions, participant consent and withdrawal scope, or the permitted downstream handling of the complete records.

CC-BY-SA attribution and ShareAlike obligations must be reviewed for the exact selected content and proposed release. In particular, Lumi's intended model-weight distribution remains undecided, and this audit does not decide whether private or public model artifacts would satisfy those obligations. This is a provenance and use gate, not a legal conclusion.

Before any item is used, the project needs to:

1. Pin the exact repository commit and release files; reconcile JMRD's repository counts and split with the published-paper counts.
2. Identify source lineage for each selected dialogue, annotation, metadata field, review, and questionnaire. Confirm the rights and attribution obligations for those exact items and for the intended diagnostic use.
3. Review participant consent, privacy, pseudonymization, timestamps, questionnaires, sensitive disclosures, retention, and removal conditions. For RecomMind, minimize or remove participant metadata unless it is explicitly needed and cleared.
4. Obtain a specific ShareAlike/model-artifact compatibility review once Lumi's distribution scope is decided. Do not infer model-weight licensing from a repository badge.
5. Assign any public items only to a declared development diagnostic. Do not use public items as the sealed final holdout. Keep any admitted evaluation items out of training, tokenizer fitting, synthetic generation, filtering, and tuning data construction.
6. Have the qualified Japanese reviewer check naturalness, meaning, and any proposed labels independently. The two semantic reviewers and this separate Japanese/bilingual reviewer are available, but no annotations or source-item reviews have been completed.

## Current disposition

- JMRD: candidate for a small public Japanese dialogue diagnostic after item-level rights, privacy, version, and language review; not approved or downloaded.
- RecomMind: higher-priority candidate for Japanese preference-tracking diagnostics after the same gates; not approved or downloaded.
- Neither source can fill the English/Japanese code-switch gap, serve as a sealed final holdout, or validate ZenStream-specific state-changing intents by itself.
- No source text or item identifiers have been entered into `provenance/data_sources.json`; the training manifest remains empty.
- If contributor and upstream rights cannot be established for these uses, exclude both releases and prioritize consented Japanese/bilingual author-written evaluation cases with the existing contribution-scope and independent-review process.

## Primary sources

- [JMRD author repository](https://github.com/ku-nlp/JMRD): current README reports its file counts, notes the mismatch with reference papers, describes the data format, and declares CC-BY-SA 4.0.
- Kodama, Tanaka, and Kurohashi (2022), [JMRD paper](https://aclanthology.org/2022.dialdoc-1.9/): corpus statistics, collection design, and external-knowledge sources.
- [RecomMind author repository](https://github.com/ku-nlp/RecomMind): current README describes split files, annotations, participant metadata, movie metadata fields, and declares CC-BY-SA 4.0.
- Kodama et al. (2024), [RecomMind paper](https://aclanthology.org/2024.sicon-1.4/): corpus statistics, internal-state annotation, and movie-information collection.
- [Creative Commons Attribution-ShareAlike 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode): license terms to review alongside the exact source-item and model-distribution facts.
