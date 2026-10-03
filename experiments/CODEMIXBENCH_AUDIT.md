# CodeMixBench coverage and provenance audit

**Reviewed:** 2026-10-03

**Decision:** Research-method reference only. No CodeMixBench dataset, prompts, labels, or model outputs were downloaded, copied, admitted, or used. The audited task table contains no Japanese-English pair, so CodeMixBench is not a direct EN/JA evaluation source for Lumi.

## Scope

This review covered the [EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.109/) and the [author repository overview](https://github.com/Jeromeyluck/CodeMixBench). It did not access the linked dataset files, prompts, or generated outputs, and did not audit the underlying releases named by the paper.

## Findings

- The paper describes eight tasks across 18 languages and presents CodeMixBench as a broad code-mixing benchmark. Its appendix Table 3 enumerates 16 language pairs; none includes Japanese. The repository overview's listed 12-language subset also omits Japanese. In particular, the benchmark does not supply an English-Japanese pair in the audited task coverage.
- The benchmark combines adapted language-identification, part-of-speech, named-entity, sentiment, and machine-translation datasets with synthetic multiple-choice, math, and truthfulness sets. The paper describes word substitution plus GPT-4 prompting for synthetic code-mixed text. This supports treating synthetic construction as a separate experimental factor and requiring human language-quality review; it does not establish naturalness for Lumi's target use.
- The tasks measure general code-mixing and reasoning. They do not measure ZenStream media intent, arguments and constraints, clarification, or safe action boundaries.
- The paper points to public code and data, but the reviewed article and repository overview do not establish item-level rights, exact source revisions, annotation grants, privacy conditions, or a dataset license for Lumi's proposed uses and weight distribution. Public repository visibility and the paper's publication terms are not treated as dataset permission.

## Decision for Lumi

Use the paper only as general benchmark-design and synthetic-data methodology context. Do not use its text, prompts, labels, generated outputs, or underlying source datasets for training, tokenizer fitting, development evaluation, or holdout. Retain the natural EN/JA evaluation gap; the separately audited [CSR-L Japanese-English query lead](../DATASET_CANDIDATES.md) remains unresolved and is retrieval-specific.

## Sources

- Yang and Chai, [CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages](https://aclanthology.org/2025.emnlp-main.109/), EMNLP 2025, especially Appendix A/Table 3 and Appendix B.
- [Official CodeMixBench repository](https://github.com/Jeromeyluck/CodeMixBench), overview page, reviewed 2026-10-03.
