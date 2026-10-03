# Evaluation sample-size plan

**Status:** Planning only. There are no ready cases, completed human reviews, candidate predictions, or final holdout measurements. The current 101-case draft is not eligible for review or scoring and is far below the statistical floor for the strictest existing target even if every case were usable and every prediction were correct.

## What the counts mean

For a target success rate `T`, if all `n` independent scored trials are correct, the one-sided 95% exact Clopper-Pearson lower confidence bound is `0.05^(1/n)`. The smallest all-correct sample that can put that lower bound at or above `T` is:

```text
n = ceil(log(0.05) / log(T))
```

This is a best-case floor, not a recommended total dataset size, guaranteed power calculation, or proof of semantic coverage. Any observed errors require a larger sample to establish the same lower bound. Compute the exact bound from the actual numerator and denominator for every reported result.

| Existing target | All-correct independent trials needed for a one-sided 95% lower bound to reach the target |
|---|---:|
| Structured-response validity >= 99.9% | 2,995 |
| Simple action selection >= 97% | 99 |
| Argument extraction >= 95% | 59 |
| Negation/no-action correctness >= 98% | 149 |
| English tasks >= 97% | 99 |
| Japanese tasks >= 95% | 59 |
| English/Japanese code-switching, aggregate >= 94% | 49 |
| Multi-turn reference resolution >= 92% | 36 |
| Difficult ambiguous requests >= 90% | 29 |

These are marginal per-metric 95% bounds. They do not provide a 95% simultaneous guarantee across all metrics and slices. Before opening the final holdout, preregister how multiple metrics will be interpreted and whether any multiplicity adjustment is required. Do not retroactively choose the rule that makes a candidate pass.

The existing plan requires reporting the two code-switch matrix directions separately, but it currently sets only an aggregate 94% acceptance threshold. Keep directional counts and intervals visible and define any separate directional pass thresholds before the confirmatory holdout; do not imply that the aggregate threshold establishes both directions independently. The state-changing and unclassified-action rates remain `not_measured` until the capability registry is approved and thresholds are frozen.

## Independent cases and slice denominators

The formula assumes independent Bernoulli trials. Count distinct semantic families, not paraphrase rows, as the conservative independent unit for planning. Keep close variants, translations, title substitutions, and related turns in the same family and the same split. If multiple cases from one family are scored, report case counts but use family-aware uncertainty estimates, such as a preregistered cluster bootstrap; do not report the simple binomial bound as if those cases were independent.

Each metric uses its own eligible denominator. For example, argument extraction uses cases with required arguments; multi-turn resolution uses cases that require a prior-turn reference; and difficult ambiguity uses cases tagged for that slice. A large overall dataset does not establish a smaller slice target. Report numerator, denominator, error count, interval method, and family count for every metric.

The table covers each existing aggregate threshold only. There is no numeric false-action-rate target yet. False actions must still be counted separately from missed actions, and the no-action/negation slice must include respond, clarify, and read-only cases as required by the scorer contract. Do not claim a low false-action rate from aggregate accuracy or from an unapproved action registry.

## Development and final holdout

Use development cases for candidate debugging and iteration; their scores and confidence intervals are selection evidence, not an unbiased final quality claim after repeated tuning. Freeze a separate, source- and family-separated final holdout with enough eligible independent families for the claims planned before the final run. Score the selected candidate once under the preregistered analysis. Never use holdout text, labels, failure examples, or reviewer notes for tokenizer fitting, training, synthetic generation, filtering, architecture choices, or hyperparameter selection.

The sample-size floor is only one gate. Both splits still require rights-approved sample provenance, adequate English/Japanese/two-direction code-switch coverage, at least two independent semantic annotations per case, a separate qualified Japanese/bilingual language review where applicable, privacy review, and adjudication. Candidate responses also require their own review for language, naturalness, meaning preservation, intent appropriateness, and concision. The number of annotated outputs can therefore be much larger than the number of authored cases.

The current development draft has zero ready cases and no review records. Do not import it into a scorer or start model comparisons until its separate provider-terms and use-scope gate passes. If the required independent human-authored set cannot be assembled, report the target as unverified instead of weakening the threshold or treating missing scores as zero errors.
