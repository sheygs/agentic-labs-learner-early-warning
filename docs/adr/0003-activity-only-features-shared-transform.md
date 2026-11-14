# ADR-0003: Use activity-only features built by one transformation shared with the API

**Status**: Accepted
**Date**: 2026-11-14

## Context

OULAD includes demographics, prior attempts, assessment scores and final results alongside the click log. The target workflow is a different organisation's course, where the only signal certain to exist two weeks in is some record of daily activity. A model is also only as good as the match between the features it was trained on and the features it receives in production.

## Decision

Use five features derived from 14 daily click counts, and compute them with a single function for both training and serving.

- `active_days_week_1`, `active_days_week_2`
- `clicks_week_1`, `clicks_week_2`
- `days_since_last_activity` (14 for a learner with no activity)

Identity, demographics, grades and final results are excluded from the feature matrix. Learner IDs are used only to join tables and remove repeat enrolments. Final results are used only to exclude withdrawals with missing dates.

`engagement_features` in `data.py` is called by the training pipeline and by `POST /predict`. It rejects anything other than a two-dimensional array of non-negative integers with 14 columns.

## Alternatives considered

- **Add demographic attributes**: they may improve the score on OULAD, but they would rank learners partly on who they are, and would not transfer to a course that does not collect them.
- **Add assessment results**: assessment timing is specific to this course's schedule, and grades would not transfer to a course that is assessed differently.
- **Engineer features in the notebook and reimplement them in the API**: two implementations drift apart, and the resulting train/serve skew is silent.
- **Send precomputed features to the API**: callers would have to reproduce the transformation exactly.

## Consequences

**Benefits**

- The API needs only an opaque ID and 14 integers per learner. No personal data is required.
- Training and serving cannot disagree on feature definitions.
- Five features with clear meanings keep the model inspectable.

**Costs**

- Predictive signal is limited. On the test intake, the model matched a one-feature recency rule at ten check-ins.
- Clicks are a proxy for engagement. They miss offline study, and a logging gap looks the same as inactivity.
- Excluding demographics from the inputs does not show that the ranking is equally accurate across groups. That has not been measured.

## Implementation

`FEATURES` and `engagement_features` in [data.py](../../data.py), imported by [train.py](../../train.py) and [predict.py](../../predict.py).
