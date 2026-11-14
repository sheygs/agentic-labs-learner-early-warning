# ADR-0004: Split chronologically by intake and serve the validation winner

**Status**: Accepted
**Date**: 2026-11-14

## Context

The model will always score a future intake using what was learned from past ones. Course BBB has four intakes in the data: 2013B, 2013J, 2014B and 2014J. The withdrawal rate changes from one intake to the next, so a random split would mix past and future and overstate performance.

After tuning, the three model families are close. On the held-out intake, random forest scored a higher average precision (0.0449) than the model chosen on validation, logistic regression (0.0426).

## Decision

Split by intake in time order, select on validation only, and serve the validation winner even when another model scores higher on the test intake.

| Stage    | Intakes             | Use                                         |
| -------- | ------------------- | ------------------------------------------- |
| Train    | 2013B, 2013J        | Fit each of the 18 candidate configurations |
| Validate | 2014B               | Choose hyperparameters and the model family |
| Refit    | 2013B, 2013J, 2014B | Refit each family's best configuration      |
| Test     | 2014J               | Report once. Plays no part in any choice    |

Selection is by validation average precision, with the lower Brier score breaking exact ties. The seed is fixed at 42.

## Alternatives considered

- **Random or stratified k-fold split**: learners from the same intake would sit on both sides, and the estimate would ignore the drift between intakes.
- **Serve random forest because it won on the test intake**: the test intake would then have been used for selection, and the reported test score would no longer be an unbiased estimate.
- **Time-series cross-validation over intakes**: with four intakes there are too few folds to gain much, at the cost of a more complex procedure.

## Consequences

**Benefits**

- The test score is an honest estimate of performance on the next intake.
- The served model is the simplest of the three candidates and the easiest to inspect.
- Results for all families are still published, so the comparison is transparent.

**Costs**

- The served model is not the best performer on the latest data.
- Selection rests on one validation intake with 24 positives, so the choice between families is fragile.
- A single test intake with 30 positives gives wide uncertainty, and no confidence intervals are reported.

## Implementation

`split_cohorts`, `candidates`, `selection_key`, `tune_models` and `evaluate_models` in [modeling.py](../../modeling.py). Results are in [metrics.json](../../reports/metrics.json).
