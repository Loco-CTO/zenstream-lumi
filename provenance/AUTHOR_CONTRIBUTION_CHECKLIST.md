# Human contributor intake checklist

**Status:** Draft operational checklist only. It is not a consent agreement, privacy notice, or legal determination. No new human-authored Lumi cases have been collected under it. Do not solicit or ingest contributions until the use scope, artifact-distribution disclosure, storage process, and required legal/privacy review are settled.

## 1. Set the contribution scope before asking

Write down, in plain language, what Lumi is expected to do with the contribution and who will be able to see it. Decide each item explicitly; an unchecked or uncertain item is not permission:

- **Evaluation only:** score candidate behavior on the named development split or sealed final-holdout split. State who can access raw examples and review notes, how long they are retained, and whether anything will be published. Do not use evaluation-only material in tokenizer fitting, training, synthetic-generation prompts, filtering, or tuning.
- **Tokenizer training:** use the sample text to fit tokenizer vocabulary or normalization. This is distinct from model-weight training.
- **General pretraining:** use the exact approved text as next-token or other declared language-model training data.
- **ZenStream-specific training:** use the text, intent labels, slots, or responses to teach media-task behavior.
- **Filtering or generation:** use the material to select, transform, or prompt the creation of other examples. Declare the exact method and lineage retained for derived samples.
- **Model artifact distribution:** state whether resulting weights or packages remain private, are shared with a named group, or are downloadable by the public; name the license or access restrictions, commercial status, and downstream modification terms when known. “Local AI” describes where inference runs; it does not by itself describe who can receive the weights.
- **Attribution, credit, or compensation:** state what will be provided, if anything, before collection.

If the model-weight distribution decision is not ready, keep any author permission limited to the exact internal evaluation use that is already defined. Record training uses as ungranted and `model_artifact_distribution` as `unknown` or `not_applicable` for evaluation-only items. Do not infer that a contributor who allows evaluation also allows tokenizer training, training, generation, public release, or commercial use.

Keep final-holdout permission and handling distinct from development use. The holdout remains access-controlled and cannot be shown to candidate developers or used to tune, filter, generate, or train examples.

## 2. Tell contributors what they are agreeing to

Before collecting any sample, provide a separate, readable disclosure that identifies:

1. The specific purposes checked above and the exact text, labels, annotations, and derived material covered.
2. The intended model artifact distribution and the people or organizations who may receive it. If that choice is unresolved, do not request a training grant.
3. How attribution, credit, compensation, retention, security, and access will work.
4. A contact and process for changing a grant or requesting withdrawal, and the exact cutoff and practical limits for removing data or derived material after it has been used in training. Do not promise that a released model can be untrained or recalled unless that has been demonstrated and is contractually supportable.
5. Whether any personal information is expected. Prefer fictional, privacy-safe examples. Do not use real ZenStream user prompts, account state, private conversations, credentials, or identifying media history for this contribution set.

Have a qualified reviewer check the proposed disclosure and collection process before anyone signs. This checklist does not choose a legal basis or establish that consent is legally sufficient. ICO guidance distinguishes anonymised information from pseudonymised information and says pseudonymised information can still be personal data; its research and consent guidance is currently subject to updates. Assess the actual data and applicable obligations rather than treating pseudonyms as a substitute for privacy review. See the UK ICO guidance on [research safeguards](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/the-research-provisions/what-are-the-appropriate-safeguards/) and [obtaining and managing consent](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/lawful-basis/consent/how-should-we-obtain-record-and-manage-consent/).

## 3. Collect only material the author may contribute

- Ask contributors to write original examples specifically for Lumi. Do not solicit copied song lyrics, subtitle lines, books, scripts, private messages, or text from websites unless the exact item and rights are separately cleared.
- Do not ask anyone to reveal another person's private text, account, library, playback state, identity, or credentials. Build synthetic trusted-context fixtures from invented titles and IDs.
- Ask the author to flag any outside source, quotation, translation, generative-AI assistance, or personally identifying detail before submission. Keep an item in `pending` until those facts and any required permissions are resolved.
- Keep each source item and its derivatives traceable. Do not paraphrase an uncleared sample and assume the new wording is rights-free.
- Separate author identity and contact details from case text using an opaque author pseudonym and consent-record ID. Keep the identity mapping, signed record, and any raw intake file in controlled storage outside the Git worktree.

## 4. Record and review each item

For each item, retain the private agreement or permission record and a metadata-only provenance join containing:

- Consent-record ID, author pseudonym, source ID, exact item ID, and canonical sample SHA-256.
- The individually granted use scopes, model artifact distribution scope, grant status, granted and expiry dates, and review rationale.
- Parent item IDs and hashes for every transformed, split, or generated derivative.
- The named evaluation split, if applicable, plus collection, transformation, and privacy-review records.

Store the metadata in the controlled `author_consents.json` manifest using [its schema](author_consents.schema.json); do not check agreements, signatures, names, contact details, raw identity mappings, or contribution text into the public repository. A grant stays `pending` until a human rights reviewer confirms that the evidence covers the exact item, use, lineage, and intended model artifact distribution. The automated validator checks joins and declared scope; it does not decide legal sufficiency.

Human case review remains separate from contributor permission. Each evaluation case still needs the independent semantic annotations, the separate qualified Japanese/bilingual review where applicable, privacy checks, family grouping, split controls, and provenance validation in the [annotation guide](../evaluation/ANNOTATION_GUIDE.md). Do not make a case `ready` merely because its author granted a use.

## 5. Keep a reviewable record of decisions

Before moving any contribution into a dataset build, record:

- Why each selected source and use is permitted, including any rights limitations or attribution obligations.
- What is excluded and why; do not silently broaden the original grant to cover exclusions later.
- The manifest, sample hashes, consent-evidence hashes, transformation code/configuration, and reviewer records used for that build.
- The withdrawal/retention decision and whether any previously derived or trained artifact is affected.

Current status: the tracked consent manifest is empty, no agreement has been executed or reviewed, and the intended model-weight distribution remains undecided. The existing Codex-generated evaluation draft is governed by its separate provider-terms decision and is not made usable by this checklist.
