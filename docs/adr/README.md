# Architecture decision records

This directory records the significant design decisions behind Learner Early Warning: what was decided, why, and what it costs. Each record is short and stands alone.

## Index

| ADR                                                       | Title                                                                           | Status   | Date       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------- | -------- | ---------- |
| [0001](0001-rank-for-fixed-outreach-capacity.md)          | Rank a cohort for a fixed outreach capacity instead of classifying each learner | Accepted | 2026-11-14 |
| [0002](0002-day-14-snapshot-and-label-rules.md)           | Score at day 14 for a 30-day horizon, with explicit eligibility and label rules | Accepted | 2026-11-14 |
| [0003](0003-activity-only-features-shared-transform.md)   | Use activity-only features built by one transformation shared with the API      | Accepted | 2026-11-14 |
| [0004](0004-chronological-split-and-validation-winner.md) | Split chronologically by intake and serve the validation winner                 | Accepted | 2026-11-14 |
| [0005](0005-stateless-batch-ranking-api.md)               | Expose a stateless batch-ranking API with strict input validation               | Accepted | 2026-11-14 |
| [0006](0006-model-artifact-with-metadata-checks.md)       | Ship the model as a committed, metadata-checked joblib artifact                 | Accepted | 2026-11-14 |
| [0007](0007-locked-environment-and-hardened-container.md) | Lock the environment and run a hardened, single-replica container               | Accepted | 2026-11-14 |

## Open decisions

These questions are not yet decided or not yet justified in writing. Each should become an ADR, or an amendment to one, before the model is used outside this prototype.

| Question                                                                             | Why it matters                                                                                              |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Why course BBB, a 14-day observation window, a 30-day horizon and a capacity of ten? | These constants define the problem. Their rationale and any alternatives tried are not recorded.            |
| Should risk scores be calibrated?                                                    | Scores are used only for ordering today. Showing them to mentors as probabilities would need calibration.   |
| How is the API protected?                                                            | There is no authentication or rate limiting. The Service is cluster-internal, which is the only barrier.    |
| How is the model monitored and when is it retrained?                                 | Withdrawal rates fell from 6.5% to 1.7% across intakes. Nothing detects drift or triggers retraining.       |
| How are model versions assigned?                                                     | `model_version` is the fixed string `day14-v1` in `train.py`. A retrained model would carry the same label. |
| Does the ranking perform equally across learner groups?                              | Demographics are excluded from the features, but performance by group has not been measured.                |
| What licence covers the code?                                                        | The repository is public and has no `LICENSE` file.                                                         |

## Adding a record

1. Copy [template.md](template.md) to `NNNN-short-title.md`, using the next number.
2. Fill in every section. State the real costs under Consequences.
3. Add a row to the index above.
4. Do not rewrite an accepted record. Write a new one that supersedes it and update the old record's status.

## Statuses

- **Proposed**: under discussion
- **Accepted**: decided and implemented
- **Superseded**: replaced by a later record
- **Rejected**: considered and not adopted
