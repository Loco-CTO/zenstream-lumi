# Synthetic output terms and eligibility audit

**Reviewed:** 2026-10-03

**Status:** Current public terms reviewed; the 101-case assistant-generated draft inventory remains blocked.

This is a provenance and project-use review, not legal advice. It records current public OpenAI terms that may be relevant to the draft inventory. It does not determine which agreement governed the historical generations or whether Lumi legally competes with an OpenAI product.

## Draft inventory evidence

The metadata-only [`draft_inventory_manifest.json`](../evaluation/draft_inventory_manifest.json) records 101 development drafts across 43 semantic families. It identifies the provider as OpenAI and the generator as “Codex assistant,” but says the exact serving model ID/build is not exposed and leaves the applicable account agreement and Lumi use/release scope unresolved. The controlled inventory records no ready cases and no review records. The tracked [`synthetic_data.json`](synthetic_data.json) manifest is empty; it does not provide item-level generation records for those drafts.

The inventory’s published file and provenance hashes identify the controlled artifacts without copying their text into this repository. A hash establishes artifact identity, not permission to reuse the content.

## Current public terms checked

The [Europe Terms of Use](https://openai.com/policies/eu-terms-of-use/), updated January 16, 2026, say they apply to ChatGPT and other OpenAI services for individuals when a person resides in the UK, EEA, or Switzerland. They prohibit using Output to develop models that compete with OpenAI. The same terms say the user owns Output as between the user and OpenAI, while also warning that outputs may not be unique. The terms do not decide whether Lumi falls within the competition restriction.

The [OpenAI Services Agreement](https://openai.com/policies/services-agreement/), effective January 1, 2026, applies to APIs, ChatGPT Business, ChatGPT Enterprise, and other business/developer services. It likewise restricts use of Output to develop AI models that compete with OpenAI, except for a defined “Permitted Exception.” That exception covers models primarily intended to categorize, classify, or organize data only when they are not distributed or made commercially available to third parties, plus fine-tuning or customization of models provided by OpenAI. The agreement assigns Output to the customer but also states that outputs may not be unique. Lumi’s planned language understanding and intent interpretation may overlap with classification, but the project also includes natural-language interaction and its weight-distribution target is undecided. The exception cannot be assumed to cover Lumi or the draft data.

The [OpenAI Service Terms](https://openai.com/policies/service-terms/) add conditions for particular products and features. The Terms of Use and Services Agreement also refer to service-specific terms and other applicable agreements. Therefore, the current public pages are not a substitute for identifying the exact account/workspace, service, agreement, order form, and terms version that applied when each draft was generated.

## Project decision

Keep the 101 drafts out of human annotation, candidate scoring, training, tokenizer fitting, synthetic-data prompts, filtering, and architecture or hyperparameter selection until the applicable terms and intended use are reviewed. This conservative gate does not conclude that Lumi is a competing model; it avoids treating that unresolved question as permission. The decision applies to every derivative of the drafts as well as the original text.

Output ownership alone is insufficient evidence for reuse. The inventory still needs a source-level and item-level generation record, a rights/use-scope review for each intended purpose, privacy review, and checks for non-unique or third-party material before any item can be admitted. Evaluation-only use, reviewer access, tokenizer fitting, training, and model-weight distribution are distinct purposes and must be assessed separately.

## Evidence needed to revisit

1. Confirm whether generation used an individual Codex/ChatGPT account, an organization-managed workspace, an API account, or another service; identify the applicable agreement and any service-specific terms.
2. Record each generation date, available model/version identifier, prompt/spec revision, and any filtering or editing that affected the 101 drafts. Do not invent missing fields; mark unavailable fields explicitly.
3. Resolve whether Lumi’s product scope and proposed use of the output fall within the relevant model-development restriction or exception. Review public and commercial distribution separately from internal use.
4. Decide the weight-distribution target before using any exception whose condition depends on whether a model is distributed or commercially available.
5. If use is approved, update the controlled source and item records with the exact permitted purposes and evidence, then rerun the existing provenance gate before exposing any draft to reviewers or the evaluation harness.

## Sources

- OpenAI, [Europe Terms of Use](https://openai.com/policies/eu-terms-of-use/), updated 2026-01-16; accessed 2026-10-03.
- OpenAI, [OpenAI Services Agreement](https://openai.com/policies/services-agreement/), effective 2026-01-01, updated 2025-12-01; accessed 2026-10-03.
- OpenAI, [Service Terms](https://openai.com/policies/service-terms/); accessed 2026-10-03.
- OpenAI Help Center, [Using Codex with your ChatGPT plan](https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan); referenced by the existing draft inventory metadata, accessed 2026-10-03.
