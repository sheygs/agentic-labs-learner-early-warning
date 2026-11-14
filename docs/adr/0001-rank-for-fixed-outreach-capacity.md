# ADR-0001: Rank a cohort for a fixed outreach capacity instead of classifying each learner

**Status**: Accepted
**Date**: 2026-11-14

## Context

A mentor has a fixed number of check-in slots, ten in this prototype. The useful question is "who should I contact first?", not "will this learner withdraw?".

Withdrawals in the outcome window are rare and becoming rarer: 6.5% of eligible learners in the 2013B intake and 1.7% in 2014J. With a base rate that low, a classifier that never predicts a withdrawal scores about 98% accuracy, and any fixed probability threshold would return a different number of learners for every cohort.

## Decision

Treat the problem as ranking a cohort, and evaluate it at the mentor's capacity.

- **Selection metric**: average precision on the validation intake, which scores the whole ranking and is sensitive to rare positives.
- **Decision metric**: expected withdrawals among the top ten (`expected_hits_at_10`), reported with precision and recall at ten.
- **Tie handling**: when a score tie straddles tenth place, the remaining slots are averaged over the tied learners. The metric therefore does not depend on row order and can be fractional.
- **Reference points**: a random shortlist and two parameter-free rules (no activity in week two, days since last activity) are evaluated the same way.

## Alternatives considered

- **Binary classification with a probability threshold**: the size of the flagged group would vary by cohort and would not match a fixed number of slots.
- **Accuracy or F1 as the metric**: both depend on a threshold, and accuracy is dominated by the majority class.
- **ROC AUC as the selection metric**: it is reported, but it weights the whole ranking equally, while the mentor only acts on the top.
- **Counting hits at ten by sort order**: many learners share the same engagement pattern, so ties at the cutoff are common and the result would depend on how tied rows happen to be ordered.

## Consequences

**Benefits**

- The headline number is in the mentor's terms: withdrawals found per ten check-ins.
- Simple rules and models are compared on the same footing. On the test intake, the served model matched the recency rule at ten check-ins, and that result is reported.

**Costs**

- Scores are used for ordering only. They are not calibrated probabilities, and the Brier score is used only to break exact ties in selection.
- The capacity of ten is a constant. A team with a different capacity needs the evaluation rerun at its own value.
- With about 30 positives per intake, top-ten metrics are noisy.

## Implementation

`expected_hits_at_k`, `ranking_metrics` and `baseline_scores` in [modeling.py](../../modeling.py). `OUTREACH_CAPACITY = 10`. The tie behaviour is covered by `test_ties_do_not_depend_on_learner_order`.
