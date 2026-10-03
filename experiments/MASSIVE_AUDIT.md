# MASSIVE 1.1 source audit

**Reviewed:** 2026-10-03

**Decision:** The exact pinned source was admitted for a bounded local English/Japanese structured intent/slot pilot; no broader use is approved.

This is a source and benchmark audit plus a scoped pilot record, not a legal opinion or a Japanese-language approval. The archived files are in workspace-level controlled storage under `.lumi-data/massive/1.1/`; they are not checked into Git. Only the 256 official-train and 64 disjoint public-dev aligned ID families listed by the local selection manifest entered the structured-output pilot. No MASSIVE text entered tokenizer fitting, broad pretraining, conversational generation, or tuning.

## Pinned artifact

- Owner and repository: Amazon Science, [`alexa/massive`](https://github.com/alexa/massive).
- Version: MASSIVE 1.1. The pinned [README at commit `f966f21846043aabef9b0f974fa7970027f43738`](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/README.md) says 1.1 adds Catalan and leaves the other languages unchanged from 1.0.
- Official archive: [`amazon-massive-dataset-1.1.tar.gz`](https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz), accessed 2026-10-03.
- Archive size: 40,251,390 bytes.
- Archive SHA-256: `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
- Archive `1.1/LICENSE` SHA-256: `c2e6ea015269147de02117ebdd91f30ef09831251f5345fa8365273b1db1d435`.
- Archive `1.1/NOTICE.md` SHA-256: `b90534ccd20c6f0e1e5239567af0d150496339542b75a15bfbc3e1e737593ddb`.
- `1.1/data/en-US.jsonl`: 3,904,197 bytes; SHA-256 `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e`.
- `1.1/data/ja-JP.jsonl`: 12,263,054 bytes; SHA-256 `c22df382db6aa4a23dd1e7f62a2ac8f01c6158865771ad25201696be7201ab79`.

Each pinned locale file contains 16,521 rows: 11,514 train, 2,033 dev, and 2,974 test. The Japanese and English files have matching IDs and the same split counts. The current [Hugging Face dataset card](https://huggingface.co/datasets/AmazonScience/massive) reports 19,521 utterances per language in its summary, while its listed train/dev/test counts sum to 16,521. The ACL paper separately describes an additional 3,000-per-language competition-heldout set that was to be released after the competition. The pinned tar contains only train, dev, and test. Keep these artifact boundaries explicit; do not infer that the heldout set is present in this archive or use any public source split as Lumi's sealed final holdout.

## Rights and attribution

The pinned repository [README](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/README.md), archive `LICENSE`, and [NOTICE](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md) declare MASSIVE under CC BY 4.0. The NOTICE also identifies the English source text as SLURP and says SLURP text is CC BY 4.0. CC BY 4.0 allows reuse and adaptation, including commercial use, subject to its attribution, license-link, change-indication, and no-endorsement conditions. Preserve attribution to Amazon and the named MASSIVE and SLURP authors, the source links, the license link, and any modifications with any later distribution of covered material.

This review covers the source's stated license for the pinned archive. It does not prove rights outside that grant, determine whether trained weights reproduce protected material, or approve public distribution of Lumi weights. The item-level provenance manifest approves only the exact training/development families in the local pilot. Any future training or public model-artifact distribution needs a separate rights, attribution, and exact-artifact lineage review.

## Quality and fit

The [ACL 2023 paper](https://aclanthology.org/2023.acl-long.235/) describes the corpus as realistic English virtual-assistant utterances localized by professional translators into 50 languages. The release includes 18 scenarios, 60 intents, and 55 slot types. Its documented workflow separately translates/localizes slot values and whole phrases. Three workers judge each localized utterance for semantic intent, slot labels, grammar/naturalness, spelling, and language identification; the collection also monitored translation quality. The [dataset card](https://huggingface.co/datasets/AmazonScience/massive) describes the corpora as free of personal or sensitive information.

An archive-level review of all 2,033 Japanese dev rows and their three judgments found 151 rows with at least one grammar score below 3/4 and 242 rows with disagreement among intent scores. These counts are direct observations from the pinned `ja-JP.jsonl`; they are not new human judgments and do not mean that the other rows are Lumi-ready. The release's workers are not a substitute for Lumi's separate qualified Japanese/bilingual review.

The source can potentially support a limited comparison of language/intent/slot behavior on public, single-turn, device-directed utterances after exact sample selection and item-level provenance. It is localized from English, not spontaneous Japanese conversation. Its multilingual judgments that include English are a lead for manual inspection, not validated coverage of meaningful Japanese-English code-switching. The dataset covers generic voice-assistant tasks rather than ZenStream's authoritative catalog state, permissions, playback negotiation, multi-turn corrections, or low-false-action policy. It cannot establish Lumi's overall product quality or its confidence thresholds.

## Pilot admission and results

The audit led to one bounded local pilot using the exact family selection described above. The runner checked source and member hashes and validated all 640 item-level records before training. The tiny 18,321-parameter random-init byte-level RNN encoder-decoder produced required JSON shape for 128/128 public-dev rows, but exact intent accuracy was 0/128 and slot-value micro F1 was 0.000 (TP 0, FP 0, FN 130). It is rejected as a Lumi candidate. Training loss falling from 5.5509 to 0.6359 does not offset that failed development result. See [the reproducible pilot report](MASSIVE_INTENT_SLOT_PILOT.md) for all hashes and local measurements; no raw rows or weights are committed.

This experiment supports only a structured-output diagnostic on a small public sample. Before promoting any source item to a ready Lumi evaluation case:

1. Select exact development rows and preserve the file, item ID, locale, original split, and item hash. Do not mix the English source and Japanese localization as separate independent semantic families.
2. Keep every selected item in the public development diagnostic split. Exclude all train rows from evaluation and all public data from the final holdout.
3. Record source and item-level rights, CC BY attribution, privacy review, language labels, and any transformation in the controlled provenance records; do not copy corpus text into the public repository.
4. Obtain independent annotation and a separate qualified Japanese/bilingual naturalness and meaning review before an item becomes a ready Lumi case. Preserve disagreement and reject or revise unclear translations.
5. Map source intents into Lumi's reviewed capability registry only where the fit is exact. Do not treat a MASSIVE intent label as a ZenStream action authorization.

The source rows outside the exact approved pilot scope remain excluded. No model weights, checkpoints, tokenizers, or code from the upstream repository are admissible for Lumi's random-initialized runtime model. The pilot's trained weight artifact stays local until its exact lineage and public distribution implications are separately reviewed.

## Sources

- FitzGerald et al., [MASSIVE paper, ACL 2023](https://aclanthology.org/2023.acl-long.235/), especially §§1 and 4.1–4.4.
- Amazon Science, [MASSIVE 1.1 README](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/README.md), [NOTICE](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/NOTICE.md), and the archived `LICENSE`.
- Amazon Science, [MASSIVE dataset card](https://huggingface.co/datasets/AmazonScience/massive), accessed 2026-10-03.
- Creative Commons, [Attribution 4.0 International legal code](https://creativecommons.org/licenses/by/4.0/legalcode).
