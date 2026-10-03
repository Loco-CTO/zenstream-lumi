# Tokenizer pilot protocol

**Status:** The level-1 random-init smoke used fixed hashed Unicode character n-gram features and no learned tokenizer. No tokenizer candidate has been selected. A development-quality tokenizer comparison still requires permitted training-only text and reviewed Lumi development cases.

## Question

Which tokenizer gives the best measured balance of English, natural Japanese, and English/Japanese code-switch behavior, text coverage, context capacity, model size, and CPU cost for Lumi's bounded ZenStream task?

Tokenizer compression alone is not the selection target. TokLens reports that intrinsic metrics are associated with multilingual benchmark results but not English results after controlling for model size, and notes confounding with pretraining composition. TokSuite demonstrates a useful controlled-comparison method, but its models are pretrained and its robustness set does not include Japanese. Lumi must test the target languages and task with random-initialized candidates.

## Entry gates

Do not make development-quality tokenizer comparisons until all of these are true. Small level-1 tokenizer or feature-shape probes may use temporary text with a clearly recorded local-use scope, a predeclared budget, and disjoint exploratory train/development families; they cannot establish tokenizer quality or satisfy any release gate.

1. Every tokenizer-training item has pinned source identity, revision, content hash, and documented permission for tokenizer training. The source records and processing manifest pass `provenance/validate.py`.
2. Tokenizer-training material is drawn only from an approved training split. No development, public-test, or sealed-holdout text, labels, reviewer notes, or answer-bearing material is used to fit or filter the vocabulary.
3. The evaluation development inventory contains ready, independently reviewed English, Japanese, and both directions of English/Japanese code-switch cases, with Japanese language review complete. Any available final holdout stays separate and sealed, but completing it is not a level-2 entry gate.
4. The unapproved Codex-generated 101-case draft inventory is excluded from tokenizer fitting and all tokenizer comparisons until its separate terms gate is resolved.

If any level-2 gate is missing, report the corpus-dependent comparison as blocked and make no tokenizer-quality claim. The missing final holdout and final distribution terms do not by themselves block a scoped level-1 probe.

## Candidate set

Begin with a byte-level baseline and two learned subword families:

- Direct UTF-8 byte tokens with no learned merges as the coverage and sequence-length baseline. Complete byte coverage does not imply good language modeling.
- BPE subwords with byte fallback, trained on the approved tokenizer-training split.
- Unigram subwords with byte fallback, trained on the same approved split.

For the first small pilot, compare candidate vocabularies of 4k, 8k, and 16k entries where the implementation supports those sizes. Revisit these pilot settings against actual corpus size and the model parameter budget before running; they are comparison points, not chosen deployment values. Pin the tokenizer implementation, version, normalization, pre-tokenization, byte fallback, special tokens, and training seed for every run.

## Intrinsic and systems measurements

Use the same frozen, provenance-approved diagnostic samples for every tokenizer. Keep item-level language labels and code-switch segment spans from source metadata or human annotation; do not infer Japanese or English boundaries from Unicode script alone.

Report by English, Japanese, English-matrix switching, and Japanese-matrix switching:

- UTF-8 round-trip success under the pinned normalization policy, exact-codepoint preservation, and unknown/fallback rate.
- Tokens per Unicode scalar, bytes per token, and normalized sequence length distributions (median, p90, and p95).
- English word fertility under one pinned word-count method; for Japanese, report characters/bytes per token rather than inventing a whitespace word count.
- Cross-language parity and the percentage of labeled language-switch boundaries that fall inside a token.
- Vocabulary entries and serialized tokenizer bytes; token-embedding parameter and byte cost at each candidate model width.
- Tokenization throughput and p50/p95 latency on a pinned CPU and runtime, including cold initialization if applicable.

Publish counts and denominators. Averages must not conceal very long Japanese or mixed-script outliers.

## Controlled model comparison

Only after the entry gates pass, hold architecture, training code, approved source documents, data order, optimizer recipe, evaluation revisions, and output/scoring rules constant. Initialize every model independently from random parameters using paired recorded seeds. Do not load pretrained weights, embeddings, checkpoints, or adapters.

Run two budget views because tokenizer fertility changes both the number of tokens and the amount of source text represented:

1. **Compute-matched:** cap measured training compute to the same predeclared budget for each tokenizer; record exact tokens, source bytes, documents, optimizer updates, and wall time consumed.
2. **Source-exposure-matched:** present the same ordered source-text examples to each candidate and record the resulting token/FLOP cost. This reports the practical compute difference for equal text exposure.

Keep the non-tokenizer architecture and training hyperparameters fixed during the tokenizer ablation. Report total model parameters and embedding-table bytes for each vocabulary rather than hiding that cost. If a fixed-total-parameter comparison is later needed, run it as a separate declared experiment. Do not tune on or open the final holdout.

Score held-out language loss as a diagnostic, but select only from reviewed development behavior and measured deployment cost: action/no-action correctness, arguments, clarification, Japanese naturalness and meaning, code-switch slices, structural validity, CPU latency, artifact size, and peak memory. Use the project's fixed target thresholds; do not relax them to favor a candidate. Record confidence intervals and failure examples.

## Decision record

Use `EXPERIMENT_TEMPLATE.md` for every run and attach the tokenizer artifacts, training-data manifest revision/hash, software versions, seeds, logs, and scorer revision. State separately:

- what the published papers report;
- what Lumi actually measured;
- what remains a hypothesis;
- whether each candidate is accepted, rejected, or inconclusive and why.

Do not promote an intrinsic-metric winner to the selected tokenizer unless it improves or preserves reviewed target behavior under the measured size and CPU constraints. No run in this protocol currently exists.
