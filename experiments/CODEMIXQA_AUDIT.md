# CodeMixQA Japanese-English source audit

**Reviewed:** 2026-10-03

**Decision:** Research reference only. Do not use CodeMixQA text, labels, answers, or derivatives in Lumi training, tokenizer fitting, development evaluation, filtering, generation, or final-holdout evidence unless a later item-level review resolves the release and source-rights questions.

This is a scoped source and task-fit review, not a legal opinion. No CodeMixQA or SimpleQA Verified records were downloaded, copied, transformed, or added to Lumi data or evaluation inventories.

## Source and construction

The CodeMixQA paper describes a 64,000-row benchmark derived from 1,000 English SimpleQA Verified questions. It creates four code-switched variants per source question for each of 16 language variants using random switching, selective switching, and two grammar-forcing settings. Japanese-English is one of the variants. The paper reports GPT-5.2 with low reasoning for perturbation and describes native-speaker human ratings of naturalness and semantic accuracy on a sampled set of generated text. The paper reports that its Japanese annotator group included one fluent non-native resident of Japan and two native speakers who had grown up in Southeast Asia and attended international schools.

This is synthetic factual question-answer text, not spontaneous user conversation. Its questions and answer keys can help study whether a model preserves factual QA content through prompted language mixing, but they do not represent ZenStream media requests, action/no-action distinctions, slot extraction, multi-turn references, tool use, or authoritative catalog grounding. The benchmark is public and has a fixed test split; it cannot serve as Lumi's sealed final holdout. Human review of a sample does not establish that every generated row is natural or semantically faithful.

## Provenance and release-rights findings

- The paper says the source is SimpleQA Verified and states that CodeMixQA is released under CC BY-SA 4.0. The current Hugging Face dataset card labels the dataset CC BY 4.0. The GitHub project displays Apache-2.0 repository metadata, but that does not resolve terms for the separately hosted dataset. These declarations conflict; do not choose the least restrictive one by inference.
- The SimpleQA Verified Hugging Face card currently declares MIT. CodeMixQA's visible dataset schema includes an `original_index` field, but this review did not pin a CodeMixQA file revision, map those indices against a pinned SimpleQA Verified artifact, or verify row-level transformations. Item-level source lineage therefore remains unverified for Lumi.
- The paper describes the release as open source while specifying ShareAlike conditions. This audit does not resolve how those terms apply to a Lumi checkpoint or what rights cover the underlying source questions and generated variants. Public availability is not approval for public Lumi weights.
- No exact Hugging Face dataset revision, file checksum, per-row attribution record, or complete source-to-variant manifest was pinned for Lumi. The repository's generation script also accepts an OpenAI API key; the released benchmark's paper identifies GPT-5.2 for perturbation, but it does not give the complete item-level generator provenance required by Lumi.

## Decision for Lumi

Keep CodeMixQA as a research citation only. Do not use its public samples as a substitute for human-authored, independently reviewed English/Japanese media-assistant cases. Its human ratings can inform annotation dimensions, especially separating language naturalness from meaning preservation, but they do not count as Lumi reviewer records or evaluation scores.

Separately permitted ASDC (one exact item), MASSIVE (the exact pilot cohort), and JMultiWOZ (the exact pilot cohorts) remain approved only within their recorded local diagnostics. None fills the natural English/Japanese code-switch gap. Any future CodeMixQA reconsideration requires a pinned artifact, reconciled data license, item-level source and generator lineage, intended-use review for training versus evaluation, privacy/content checks, and a new decision before any records enter a Lumi workflow.

## Sources

- Winata et al., [Can Large Language Models Understand, Reason About, and Generate Code-Switched Text?](https://arxiv.org/abs/2601.07153), arXiv:2601.07153, accessed 2026-10-03; see §§3–4 and appendices C–D.
- [CodeMixQA dataset card](https://huggingface.co/datasets/gentaiscool/codemixqa), accessed 2026-10-03.
- [CodeMixQA project repository](https://github.com/gentaiscool/codemixqa), accessed 2026-10-03.
- [SimpleQA Verified dataset card](https://huggingface.co/datasets/google/simpleqa-verified), accessed 2026-10-03.
