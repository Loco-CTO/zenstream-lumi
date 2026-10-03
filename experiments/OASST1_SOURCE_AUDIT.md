# OASST1 source audit

**Reviewed:** 2026-10-03
**Decision:** Candidate only. No OASST1 text was downloaded or admitted by this audit. The published license and upstream paper make it worth evaluating as a small general-dialogue training source, but the ready subset still needs a pinned row-level screening pass before training or public-artifact use.

## Pinned release and evidence

- Dataset: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b), repository revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, as reported by the Hugging Face dataset API on 2026-10-03.
- Card: [README at the pinned revision](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/README.md). Hub metadata declares Apache-2.0. The card describes volunteer-created, human-annotated conversation trees and the release cutoff of 2023-04-12.
- Paper: Köpf et al., [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327). The authors describe a crowd-sourced human conversation corpus and say the released data and code use a permissive license.
- The card reports 161,443 messages across 66,497 trees and 35 languages, with 1,018 Japanese messages in the full release. These counts include more than the ready-for-export parquet split; the card reports 88,838 ready messages represented by 84,437 train rows and 4,401 validation rows. The 1,018 Japanese count must not be treated as the Japanese count in those 88,838 ready rows.
- The pinned parquet objects are `data/train-00000-of-00001-b42a775f407cee45.parquet` (SHA-256 `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e`, 39,516,251 bytes) and `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` (SHA-256 `24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416`, 2,080,179 bytes). Recheck these hashes after download and reject the input on mismatch.

## Fit and limits

OASST1 may provide a small amount of broad English and Japanese assistant-style dialogue for a bounded local random-initialization training diagnostic. It is not task-matched ZenStream data. It has no demonstrated coverage for music or video intents, authoritative catalog grounding, false-action safety, English/Japanese code-switching, or the required natural Japanese response quality. Its public and old messages cannot serve as a sealed final holdout. The Japanese volume is especially small, and the actual count after the filters below is unknown.

The card schema includes `synthetic` and `model_name`, and its example shows an assistant message marked synthetic. Therefore the description “human-generated” does not establish that every exported assistant message is human-authored. Do not use any tree with a synthetic message, a non-empty generator identity, a missing/unknown synthetic flag, or unresolved authorship anywhere in its ancestry or descendants. Excluding a whole `message_tree_id` keeps generated content out of both targets and context.

The official parquet train/validation counts do not establish that all messages from one tree are assigned to the same split. Treat `message_tree_id` as the split family, combine only the pinned public ready-export rows for inventory, and make a new family-disjoint training split. Do not use the published validation split as a final holdout or as evidence of uncontaminated evaluation.

## Admission gates before a training pilot

1. Download only the two pinned parquet objects to access-controlled storage outside Git, verify their recorded byte lengths and SHA-256 values, and save the fetch date and tool version in a local acquisition record.
2. Inventory without printing or committing message text. Require stable `message_id`, `parent_id`, `message_tree_id`, role, language, deletion, review, and synthetic fields. Reject malformed ancestry, duplicate IDs, unresolved synthetic status, and any tree containing generated assistant content.
3. Reconstruct complete tree families before filtering. Exclude a whole family if any included message has a positive PII, spam, language-mismatch, hate, sexual-content, threat, or other disallowed quality flag; run a conservative local identifier scan and manually inspect a controlled sample. Retain only reviewed Japanese rows whose labels and content are consistent with Japanese. Record both raw counts and every filter count without releasing source text.
4. Check exact and normalized duplicate hashes against existing Lumi training inputs and all development/final-holdout case families. Public OASST1 material is training-only for this scope and must never be used to author, filter, tune against, or score the evaluation cases.
5. Pin an item-level lineage record for every selected message, including tree ID, message ID, source revision, source hash, license evidence, attribution, filter decisions, split, and transformed-example hash. Recheck the dataset and item rights for public model-artifact distribution before training weights intended for release.
6. If all gates pass, run one bounded random-init local training diagnostic on the family-disjoint training set. Keep the run diagnostic-only: report English/Japanese counts and training/CPU measurements, then use independent, qualified human review for Japanese response quality. Do not claim natural Japanese capability or general-dialogue competence from training loss alone.

No raw messages, reviewer-facing examples, trained weights, or text-derived snippets from OASST1 belong in Git. Until the gates pass, the source remains unapproved for training, tokenizer fitting, evaluation, filtering, generation, prompts, or release evidence.
