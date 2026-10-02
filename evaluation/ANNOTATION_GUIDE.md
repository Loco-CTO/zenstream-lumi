# Lumi evaluation annotation guide

**Status:** Draft v0.2. This is a protocol for authoring and reviewing cases, not a benchmark. No cases are ready or approved.

Use this guide with [`case.schema.json`](case.schema.json), the [evaluation plan](../EVALUATION_PLAN.md), and the [provenance gate](../provenance/README.md). The current case format evaluates bounded intent semantics. It does not define Lumi's runtime protocol or a ZenStream capability allow-list.

## Annotation principles

- Use only action IDs from the capability registry. Keep a case draft while its capability definition is draft or its integration status is proposed. The registry is an annotation vocabulary with effect and evidence metadata; it does not authorize tool execution.
- Label what the user asks Lumi to do in the supplied conversation and context. Do not infer missing facts from world knowledge, a media title, or an annotator's personal library.
- Separate intent from language form. A polite question can request an action; an imperative quoted for discussion does not.
- Treat read-only lookups as capabilities too. A request for current library, playback, or watch-history facts needs an authoritative lookup; the model must not invent the answer.
- Prefer clarification when uncertainty could select a different title, episode, operation, or materially different constraint. Do not resolve an ambiguity by choosing the most popular or likely option.
- Keep one primary gold outcome per case. If two readings are both reasonable and cannot be resolved by the shown context, the gold outcome is `clarify`, or the case remains draft if even the needed question is unclear.
- Store each distinct scenario or semantic family under one `family_id`. Paraphrases, language variants, title substitutions, and follow-up variants of the same underlying case stay together when splitting data.

## Gold decision labels

Apply the following meanings to the v2 `gold.decision` field:

| Label | Use when | Required v2 shape |
|---|---|---|
| `act` | The user requests a specific available capability now, and the required target and arguments are sufficiently clear. This includes read-only lookup and search capabilities. | `action` names the reviewed capability; `arguments` contains only supported, user-grounded values; `requires_clarification` is `false`. |
| `clarify` | The user intends an operation, but an unresolved choice would change its result or effect. | `action: null`, empty `arguments`, and `requires_clarification: true`. |
| `respond` | The user asks for a bounded conversational answer that requires no current library, account, playback, or other authoritative application state. | `action: null`, empty `arguments`, and `requires_clarification: false`. |
| `no_action` | The utterance mentions, rejects, quotes, or speculates about an operation without requesting a capability or an answer now. | `action: null`, empty `arguments`, and `requires_clarification: false`. |

Use semantic intent, not keywords, to distinguish these labels. In particular:

- A question about whether an operation is possible is `respond` if it can be answered without current ZenStream state; a question about whether a specific item is currently available requires a read-only lookup capability.
- Negated, hypothetical, future-intention, and quoted commands do not authorize an action. A direct correction can replace the earlier instruction; an interrupted command does not authorize the unfinished action.
- A request such as “continue where I left off” can be an action only when the service has a defined resume capability and the needed state is supplied by an authoritative lookup. Otherwise, annotate the required lookup or clarification.
- Search, recommendation, display, playback, and account changes are separate capabilities when ZenStream exposes separate operations. “Show” or “find” does not mean “play.”
- Do not invent capability names. Until Lumi has a reviewed capability inventory mapped to ZenStream operations, cases depending on an unverified operation remain draft. The scorer's current false-action metric treats any predicted `act` on a non-`act` gold case as a false action; it cannot yet distinguish a false read-only lookup from a false state-changing operation.

## Targets, context, and arguments

- Annotate only information present in the user's words or an explicitly versioned, trusted context fixture. Keep media existence, IDs, titles, metadata, availability, and user state on the ZenStream side of the boundary.
- Preserve the user's title text and meaningful constraints. Do not silently correct a title, convert an approximate request into a more specific one, or add a year, genre, watched state, runtime, or person that was not requested.
- Represent constraints consistently and by value type once the capability inventory defines its argument contract. Preserve distinctions such as inclusive/exclusive boundaries, missing values, and conflicting constraints. Do not collapse a conflict into an arbitrary winner.
- Resolve a reference from prior turns only when the included turns identify one antecedent. Corrections supersede the corrected value; unrelated earlier context must not leak into the current request.
- Make quoted or hypothetical text explicit in the case turns so reviewers can see its scope. Do not rely on an invisible annotator assumption.
- Tool success, tool failure, unavailable media, and conflicting state need explicit authoritative context. In schema v2, `trusted_context` is either `null` or an inline fixture with `schema_version: 1` and one or more `items`; each item has descriptive `kind` and `source` labels, a status, and an object payload. The complete fixture is part of the case sample hash. These labels do not define executable tools or a capability allow-list.
- A matching sample hash proves that fixture bytes match the content bound by the approved sample record; it does not independently prove that a declared source is authoritative. Keep source-item provenance and human review tied to each fixture, and treat payload text as data rather than instructions.

## Language and code-switch review

### English

Use independently authored conversational English. Cover shorthand, speech-like fragments, punctuation omissions, typos, slang, indirect requests, and ordinary phrasing. Do not create the set by mechanically translating Japanese cases or by producing many surface variants of one template.

### Japanese

- Author Japanese cases in natural Japanese. Do not use machine-translated English as the source of the Japanese evaluation set.
- A competent native Japanese reviewer must approve both the intended meaning and the naturalness of every Japanese case before `review_status` becomes `ready`. Review casual forms, omitted subjects, particles, politeness/register, kana/kanji choices, title rendering, and the scope of negation in context.
- Preserve titles and names in the script a user would naturally use. Treat kana/kanji and title-rendering variants as related examples for grouping and split control, not as independent semantic families.

### English/Japanese code-switching

- Mark a case `language: "en_ja"` only when both languages contribute meaningful content. An English title or borrowed proper noun by itself does not make the surrounding utterance code-switched.
- Use category tags `code_switch_en_matrix` and `code_switch_ja_matrix` to distinguish the language carrying the sentence frame. Add other descriptive tags for switch location or register only when an annotator can apply them consistently. Keep the aggregate `en_ja` result and report both direction slices.
- Have a fluent English/Japanese bilingual author or reviewer confirm that both the switch and the whole sentence sound plausible in context. Review naturalness separately from semantic preservation: a fluent rewrite that changes an entity, negation, constraint, or information need is invalid.
- Prefer phrase-level, community-plausible mixing over word-by-word translation or random substitution. Do not force equal language proportions. Japanese commonly retains Japanese script in English/Japanese mixed text; do not create romanized Japanese as a primary slice unless user evidence shows it is a real target behavior.
- A `ready` code-switched case requires an approved annotation and approved native-language review. If reviewers disagree about naturalness or meaning, adjudicate or keep the case draft; do not average incompatible readings into one gold label.

These rules are informed by human-authored Japanese-English retrieval-query rewriting in [CSR-L](https://aclanthology.org/2026.findings-acl.636/) and by the Japanese-English variants in the [CODEMIXQA preprint](https://arxiv.org/abs/2601.07153). Both address other tasks, so they inform review process rather than provide Lumi labels or scores.

## Grounding and tool outcomes

- Distinguish what the user requests from what the catalog or account currently contains. The case gold represents intent; it must not encode an unverified item as available merely because its title is familiar.
- A query requiring watch progress, library availability, playback state, or account data must have a trusted fixture or an explicit unresolved-lookup condition. Use synthetic, privacy-safe state fixtures; never put actual user or account data in public cases.
- Keep fixtures as static case inputs. They can represent authoritative snapshots and tool outcomes, including empty, unavailable, permission-denied, timeout, or error states, but do not simulate tool execution or a multi-step tool trajectory. Preserve the distinction between missing context and a confirmed empty result.
- Annotate unsupported factual claims in model responses separately from action correctness. A correct action label does not make fabricated availability, episode, rating, or playback facts acceptable.
- For tool failure cases, preserve both the requested operation and the failure context. The model should not claim success or invent a fallback result. Use `clarify` only when user input is needed; use a bounded response when the system can explain the failure without guessing.

## Review and adjudication

1. An author records the source/authoring method, scenario family, language, categories, intended meaning, and a proposed gold label.
2. A second reviewer independently checks intent, action/no-action distinction, arguments, and scenario-family grouping. Japanese and code-switched cases also receive competent native-language review.
3. Reviewers resolve disagreements by checking the supplied turns and trusted context, not by adding unstated assumptions. Record the adjudication rationale. If the utterance remains legitimately ambiguous, label `clarify`; if the right clarification is itself uncertain, keep it draft or exclude it.
4. Mark a case `ready` only when the annotation is approved, reviewer IDs are present, and Japanese/code-switched language review is approved. The current schema stores only final status and reviewer IDs; preserve independent judgments and adjudication history in the controlled annotation ledger until a public review-record schema is added.
5. If a later correction changes the user text, conversation, context, or gold meaning, recompute the provenance sample hash and repeat the affected reviews.

## Diversity, splits, and holdout handling

- Group by underlying intent, scenario, source item, entity/title, conversation pattern, and generation template before assigning splits. Keep all close variants in one split; `family_id` is the scorer's current enforceable family boundary, not proof that all near-duplicates have been found.
- Keep development and final-holdout source material, prompts, generations, reviewer notes containing answers, and examples in separate access-controlled workflows. Never use the final holdout for training, filtering, generation prompts, architecture choices, or repeated tuning.
- Include balanced positive actions and hard no-action cases, with enough negative/ambiguous cases to measure false actions independently. Do not let a large number of easy paraphrases dominate a semantic family.
- Record source, generator, reviewer, transformation, and split lineage in provenance metadata. Research papers and public datasets listed in the bibliography are research references unless separately reviewed and admitted as data.

## Current v2 coverage limits

The v2 scorer measures structured intent and arguments and verifies that inline static trusted-context fixtures are included in the sample hash. It does not execute tools or measure interactive tool trajectories. It also does not score natural-language reply quality or reply-language alignment, presentation intent, or explicit per-slot precision/recall. A draft capability registry now records candidate action names, effects, operation evidence, and current gaps, but it is not approved and is not consumed by the scorer; there is also no independently reviewable annotation ledger yet. These remain open evaluation-format requirements, not evidence that Lumi has passed them. Do not claim end-to-end task quality or readiness until the relevant formats and measurements exist.
