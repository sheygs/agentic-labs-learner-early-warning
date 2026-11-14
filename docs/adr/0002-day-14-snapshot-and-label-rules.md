# ADR-0002: Score at day 14 for a 30-day horizon, with explicit eligibility and label rules

**Status**: Accepted
**Date**: 2026-11-14

## Context

An early-warning score is only valid if everything it uses was known at the moment of scoring, and if every label reflects an outcome that was actually observed. OULAD records registration and withdrawal as day offsets from the course start, some of them missing, and some learners take the same course more than once.

## Decision

Build one snapshot per learner at the start of course day 14, and label it with withdrawal in the following 30 days.

| Rule               | Definition                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Observation window | Days 0–13. Clicks outside this range never reach the features.                                    |
| Eligible           | Registered by day 13, with a known registration date, and not withdrawn before day 14.            |
| Positive           | Withdrawal recorded on days 14–43, inclusive.                                                     |
| Negative           | Still enrolled through day 43, including learners who withdraw later.                             |
| Excluded           | Final result "Withdrawn" with no withdrawal date. The timing is unknown, so the label is unknown. |
| Repeat enrolments  | Keep each learner's earliest eligible enrolment. Later ones are dropped.                          |
| Horizon check      | Training fails if any selected intake is shorter than the outcome window.                         |

Every exclusion is counted per intake and written to `reports/metrics.json`, so the path from raw registrations to modelled rows can be audited.

## Alternatives considered

- **Label undated withdrawals as negative**: this would put real withdrawals in the negative class.
- **Label undated withdrawals as positive**: the withdrawal may have happened before day 14 or after day 43.
- **Keep every enrolment of a repeat learner**: the same person could appear in both training and test intakes, leaking information across the split.
- **Predict eventual withdrawal at any time**: a mentor acts on near-term risk, and later intakes would need the full course length before they could be labelled.

## Consequences

**Benefits**

- No feature uses information from after the scoring point.
- The negative class contains only learners known to be enrolled through the window.
- Each learner contributes to exactly one side of the chronological split.

**Costs**

- Excluded rows shrink an already small positive class: 12 undated withdrawals and 53 repeat enrolments were removed from the test intake.
- A learner who withdraws on day 44 counts as a negative.
- The observation length, horizon and course are constants in `data.py`. The rationale for 14 days, 30 days and course BBB is not recorded; see the open decisions in the [index](README.md#open-decisions).

## Implementation

`enrolment_snapshot`, `read_daily_clicks` and `build_dataset` in [data.py](../../data.py); `split_cohorts` in [modeling.py](../../modeling.py). Covered by `test_followup_window_and_no_future_clicks` and `test_retakes_do_not_cross_intakes`.
