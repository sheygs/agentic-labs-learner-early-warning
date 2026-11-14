"""Build one day-14 engagement snapshot per eligible course enrolment."""

from pathlib import Path

import numpy as np
import pandas as pd

MODULE = "BBB"
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")
DAYS_PER_WEEK = 7
OBSERVATION_DAYS = 14
HORIZON_DAYS = 30
FIRST_TARGET_DAY = OBSERVATION_DAYS
LAST_TARGET_DAY = FIRST_TARGET_DAY + HORIZON_DAYS - 1
KEY = ["code_module", "code_presentation", "id_student"]
FEATURES = [
    "active_days_week_1",
    "active_days_week_2",
    "clicks_week_1",
    "clicks_week_2",
    "days_since_last_activity",
]
TARGET = "withdrew_in_30_days"


def engagement_features(daily_clicks):
    """Use the same daily count transformation for training and API requests."""
    clicks = np.asarray(daily_clicks, dtype=np.int64)
    if clicks.ndim != 2 or clicks.shape[1] != OBSERVATION_DAYS or (clicks < 0).any():
        raise ValueError(
            f"Provide {OBSERVATION_DAYS} non-negative daily counts per learner."
        )

    active = clicks > 0
    latest_active_day = np.where(active, np.arange(1, OBSERVATION_DAYS + 1), 0).max(
        axis=1
    )
    return pd.DataFrame(
        {
            FEATURES[0]: active[:, :DAYS_PER_WEEK].sum(axis=1),
            FEATURES[1]: active[:, DAYS_PER_WEEK:].sum(axis=1),
            FEATURES[2]: clicks[:, :DAYS_PER_WEEK].sum(axis=1),
            FEATURES[3]: clicks[:, DAYS_PER_WEEK:].sum(axis=1),
            FEATURES[4]: OBSERVATION_DAYS - latest_active_day,
        }
    )


def enrolment_snapshot(data_dir):
    """Apply the observation cutoff and generate labels using withdrawal dates."""
    registrations = pd.read_csv(
        data_dir / "studentRegistration.csv",
        usecols=KEY + ["date_registration", "date_unregistration"],
        na_values=["?"],
    )
    outcomes = pd.read_csv(
        data_dir / "studentInfo.csv", usecols=KEY + ["final_result"], na_values=["?"]
    )
    outcomes = outcomes.loc[outcomes["code_module"].eq(MODULE)]
    registrations = registrations.loc[registrations["code_module"].eq(MODULE)]
    records = registrations.merge(outcomes, on=KEY, how="left", validate="one_to_one")
    if records["final_result"].isna().any():
        raise ValueError("A registration is missing its final-result record.")

    registered = records["date_registration"]
    withdrawn = records["date_unregistration"]
    records["status"] = np.select(
        [
            registered.isna(),
            registered.ge(FIRST_TARGET_DAY),
            withdrawn.lt(FIRST_TARGET_DAY),
            withdrawn.isna() & records["final_result"].eq("Withdrawn"),
        ],
        [
            "missing_registration_date",
            "registered_after_cutoff",
            "already_withdrawn",
            "missing_withdrawal_date",
        ],
        default="eligible",
    )
    audit = (
        records.groupby(["code_presentation", "status"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            index=PRESENTATIONS,
            columns=[
                "eligible",
                "missing_registration_date",
                "registered_after_cutoff",
                "already_withdrawn",
                "missing_withdrawal_date",
            ],
            fill_value=0,
        )
    )
    audit.insert(0, "raw_registrations", audit.sum(axis=1))
    eligible = records.loc[
        records["status"].eq("eligible"), KEY + ["date_unregistration"]
    ].copy()
    eligible[TARGET] = (
        eligible["date_unregistration"]
        .between(FIRST_TARGET_DAY, LAST_TARGET_DAY)
        .astype("int8")
    )
    return eligible.set_index(KEY)[[TARGET]].sort_index(), audit


def read_daily_clicks(data_dir, eligible_index):
    """Aggregate only days 0–13, with zero log activity for unmatched days."""
    totals = []
    for chunk in pd.read_csv(
        data_dir / "studentVle.csv",
        usecols=KEY + ["date", "sum_click"],
        dtype={"id_student": "int32", "date": "int16", "sum_click": "int32"},
        chunksize=250_000,
    ):
        selected = chunk.loc[
            chunk["code_module"].eq(MODULE)
            & chunk["date"].between(0, OBSERVATION_DAYS - 1)
        ]
        if not selected.empty:
            totals.append(
                selected.groupby(KEY + ["date"], sort=False)["sum_click"].sum()
            )
    if not totals:
        raise ValueError(
            f"No activity found for OULAD course {MODULE} in the observation window."
        )

    daily = (
        pd.concat(totals)
        .groupby(level=KEY + ["date"], sort=False)
        .sum()
        .unstack("date", fill_value=0)
    )
    return daily.reindex(
        index=eligible_index, columns=range(OBSERVATION_DAYS), fill_value=0
    )


def build_dataset(data_dir):
    data_dir = Path(data_dir)
    courses = pd.read_csv(data_dir / "courses.csv")
    durations = (
        courses.loc[courses["code_module"].eq(MODULE)]
        .set_index("code_presentation")["module_presentation_length"]
        .reindex(PRESENTATIONS)
    )
    if durations.isna().any() or durations.lt(LAST_TARGET_DAY).any():
        raise ValueError(
            "Every selected course intake must cover the full prediction horizon."
        )

    eligible, audit = enrolment_snapshot(data_dir)
    daily = read_daily_clicks(data_dir, eligible.index)
    features = engagement_features(daily.to_numpy())
    features.index = eligible.index
    dataset = eligible.join(features).reset_index().drop(columns="code_module")
    by_intake = dataset.groupby("code_presentation")
    audit["withdrawn_in_window"] = by_intake[TARGET].sum()
    audit["zero_activity"] = (
        dataset.assign(
            zero_activity=dataset["days_since_last_activity"].eq(OBSERVATION_DAYS),
        )
        .groupby("code_presentation")["zero_activity"]
        .sum()
    )
    audit["withdrawal_rate"] = audit["withdrawn_in_window"] / audit["eligible"]
    return dataset, audit
