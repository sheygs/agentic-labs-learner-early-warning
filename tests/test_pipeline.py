"""Regression checks for the time cutoff, cohort split and tied rankings."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data import (
    FEATURES,
    KEY,
    TARGET,
    engagement_features,
    enrolment_snapshot,
    read_daily_clicks,
)
from modeling import expected_hits_at_k, split_cohorts


class SnapshotTests(unittest.TestCase):
    def test_followup_window_and_no_future_clicks(self):
        exits = [13, 14, 43, 44, None, None, 20, 20]
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            keys = pd.DataFrame(
                [("BBB", "2014J", learner_id) for learner_id in range(1, 9)],
                columns=KEY,
            )
            outcomes = ["Withdrawn"] * 4 + ["Pass"] + ["Withdrawn"] * 3
            keys.assign(
                date_registration=[0, 0, 0, 0, 0, 0, 14, None],
                date_unregistration=exits,
            ).to_csv(directory / "studentRegistration.csv", index=False, na_rep="?")
            keys.assign(final_result=outcomes).to_csv(
                directory / "studentInfo.csv", index=False
            )
            pd.DataFrame(
                [
                    ("BBB", "2014J", 2, -1, 400),
                    ("BBB", "2014J", 2, 0, 6),
                    ("BBB", "2014J", 2, 0, 4),
                    ("BBB", "2014J", 2, 13, 2),
                    ("BBB", "2014J", 2, 14, 999),
                ],
                columns=KEY + ["date", "sum_click"],
            ).to_csv(directory / "studentVle.csv", index=False)
            labels, audit = enrolment_snapshot(directory)
            daily = read_daily_clicks(directory, labels.index)
            features = engagement_features(daily.to_numpy())

        self.assertEqual(
            labels.index.get_level_values("id_student").tolist(), [2, 3, 4, 5]
        )
        self.assertEqual(labels[TARGET].tolist(), [1, 1, 0, 0])
        self.assertEqual(features[FEATURES].iloc[0].tolist(), [1, 1, 10, 2, 0])
        self.assertEqual(features[FEATURES].iloc[-1].tolist(), [0, 0, 0, 0, 14])
        self.assertEqual(audit.loc["2014J", "missing_withdrawal_date"], 1)

    def test_retakes_do_not_cross_intakes(self):
        rows = pd.DataFrame(
            [
                ("2013B", 1, 1),
                ("2013B", 2, 0),
                ("2013J", 1, 1),
                ("2013J", 3, 0),
                ("2014B", 2, 0),
                ("2014B", 4, 1),
                ("2014B", 5, 0),
                ("2014J", 4, 0),
                ("2014J", 6, 1),
                ("2014J", 7, 0),
            ],
            columns=["code_presentation", "id_student", TARGET],
        )
        split = split_cohorts(rows)
        self.assertEqual(set(split.train.id_student), {1, 2, 3})
        self.assertEqual(set(split.validation.id_student), {4, 5})
        self.assertEqual(set(split.test.id_student), {6, 7})

    def test_ties_do_not_depend_on_learner_order(self):
        self.assertAlmostEqual(
            expected_hits_at_k([1, 0, 1, 0], [1, 0.5, 0.5, 0], k=2), 1.5
        )
        self.assertAlmostEqual(
            expected_hits_at_k([1, 1, 0, 0], [1, 0.5, 0.5, 0], k=2), 1.5
        )


if __name__ == "__main__":
    unittest.main()
