# OASST1 source audit

**Reviewed:** 2026-10-03
**Decision:** Candidate only. The pinned shards were downloaded to controlled `.lumi-data` storage and read for aggregate source screening. No message text has been admitted to or used by Lumi training or evaluation. The published license and upstream paper make this worth considering as a small general-dialogue training source, but privacy, item-lineage, overlap, Japanese-quality, and public-artifact gates remain open.

## Pinned release and evidence

- Dataset: [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404aff25b6603abec9e3399b), repository revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`, as reported by the Hugging Face dataset API on 2026-10-03.
- Card: [README at the pinned revision](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/README.md). Hub metadata declares Apache-2.0. The card describes volunteer-created, human-annotated conversation trees and the release cutoff of 2023-04-12.
- Paper: Köpf et al., [OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327). The authors describe a crowd-sourced human conversation corpus and say the released data and code use a permissive license.
- The card reports 161,443 messages across 66,497 trees and 35 languages, with 1,018 Japanese messages in the full release. Its ready-for-export parquet has 88,838 messages in 84,437 train and 4,401 validation rows. The audited ready split contains 41,305 English rows (39,283 train, 2,022 validation) and 379 Japanese rows (363 train, 16 validation); the full-release language count is not the trainable ready-split count.
- The pinned parquet objects are `data/train-00000-of-00001-b42a775f407cee45.parquet` (SHA-256 `bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e`, 39,516,251 bytes) and `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` (SHA-256 `24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416`, 2,080,179 bytes). Recheck these hashes after download and reject the input on mismatch.
- The local aggregate-only audit uses [the pinned-parquet audit script](oasst1_source_audit.py) and `pyarrow==21.0.0` from `requirements-oasst1-audit.txt`. Its controlled report is `.lumi-data/oasst1/fdf72ae0827c1cda404aff25b6603abec9e3399b/oasst1-source-audit.json`, SHA-256 `c0c8f7388d4b991a540c0c09fcd12b5df4c4419f70838cfd23870dbd7418e10f`; the script SHA-256 is `0bd0694822177e10eeb75343faba51f0b729b4126a03c3b5fd239936100a97a1`. The report contains counts and hashes only, not message text or contributor identifiers.

## Fit and limits

OASST1 may provide a small amount of broad English and Japanese assistant-style dialogue for a bounded local random-initialization training diagnostic. It is not task-matched ZenStream data. It has no demonstrated coverage for music or video intents, authoritative catalog grounding, false-action safety, English/Japanese code-switching, or the required natural Japanese response quality. Its public and old messages cannot serve as a sealed final holdout. In the published train split, only 324 Japanese rows across 45 conversation trees remain after automatic screening; that is too small to establish representative Japanese quality.

The card schema includes `synthetic` and `model_name`, and its example shows an assistant message marked synthetic. The exact ready-for-export parquet audit found zero rows marked synthetic, zero rows with an unknown synthetic flag, and zero non-empty `model_name` values. Keep the family-wide exclusion in the script so a future revision cannot silently introduce generated content. Do not use any tree with synthetic or unresolved authorship anywhere in its ancestry or descendants.

The audit found zero `message_tree_id` families crossing the official train and validation files among 10,364 ready-export trees. This resolves the family-split question for this pinned release only. The public validation split is still not a secret or Lumi final holdout and is not admitted for evaluation.

## Aggregate screen findings

- After excluding whole families with a deleted or unknown message, a non-positive or unknown review result, unsupported or mixed language, email-like text, or phone-like text, 26,796 English rows across 2,536 families and 340 Japanese rows across 47 families remain as automatic candidates. These are not training-approved examples.
- Restricting that candidate set to the published train split leaves 25,491 English rows across 2,405 families and 324 Japanese rows across 45 families. The public validation split leaves 1,305 English rows across 131 families and 16 Japanese rows across two families; it remains unused for this training-only source scope.
- The ready split has zero positive values for the screened spam, PII, language-mismatch, inappropriate, hate, sexual-content, violence, threat, identity-attack, and sexual-explicit labels. This label result does not establish that message text is PII-free.
- Conservative patterns flagged 41 families with email-like strings and 768 with phone-like strings. These reasons overlap; phone-like hits can be false positives such as technical numbers. A controlled manual privacy review is still required before any sample is admitted.
- NFKC, case-folded, whitespace-normalized matching found 443 duplicate-text groups, 288 of which span multiple conversation families. Remove or consistently assign duplicate families before any future training split, and check these hashes against approved Lumi training inputs and the frozen evaluation inventory.
- Parent-chain screening found zero missing or ambiguous parent references, zero parent/tree mismatches, zero non-alternating parent roles, and zero cycles in this pinned ready split.
- The audit script writes only aggregate counts, pinned file hashes, and a script hash. It does not emit message text, contributor IDs, message IDs, or conversation-tree IDs.

## Admission gates before a training pilot

1. Done for the pinned source: download into controlled `.lumi-data`, verify both byte lengths and SHA-256 values, and preserve the acquisition date. Keep the raw shards and report out of Git.
2. Done for aggregate structure: the script verifies required fields and row counts; checks unique message IDs, parent references, parent/tree consistency, alternating roles, and acyclic ancestry; and reports tree-family split overlap and synthetic/generator declarations without writing identifiers or text to the report.
3. Partially done: the scripted family filter removes unknown/deleted or negatively reviewed content, unsupported or mixed-language families, disallowed labels, and conservative email/phone-pattern hits. Manual review of a controlled privacy sample and qualified Japanese review remain required.
4. Pending: check source text and tree hashes against approved Lumi training inputs and the frozen development/final-holdout inventory. The family filter above was used only to screen OASST1's own candidate families. Do not use OASST1 to author, select, filter, tune, prompt, or score evaluation cases, to filter other corpora, to fit the tokenizer, or to generate synthetic content for Lumi.
5. Pending: create item-level provenance for selected samples and explicitly approve this scoped training use. Re-review the dataset and item rights for public model-artifact distribution before training weights intended for release.
6. Not started: no OASST1 training or evaluation has been run. If the remaining gates pass, keep a random-init run bounded and diagnostic; training loss alone cannot establish natural Japanese or general-dialogue competence.

No raw messages, reviewer-facing examples, trained weights, or text-derived snippets from OASST1 belong in Git. The source remains unapproved for training, tokenizer fitting, evaluation, cross-dataset filtering, generation, prompts, and release evidence until the remaining gates pass.
