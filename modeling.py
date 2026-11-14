"""Chronological splitting, model selection and ranking metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from data import FEATURES, PRESENTATIONS, TARGET

SEED = 42
OUTREACH_CAPACITY = 10


@dataclass
class CohortSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    repeat_counts: pd.Series


def split_cohorts(dataset):
    """Keep a learner's first eligible enrolment and use ordered course intakes."""
    ordered = dataset.sort_values(["code_presentation", "id_student"])
    repeats = ordered["id_student"].duplicated()
    repeat_counts = (
        ordered.loc[repeats]
        .groupby("code_presentation")
        .size()
        .reindex(PRESENTATIONS, fill_value=0)
    )
    data = ordered.loc[~repeats]
    split = CohortSplit(
        train=data.loc[data["code_presentation"].isin(PRESENTATIONS[:2])],
        validation=data.loc[data["code_presentation"].eq(PRESENTATIONS[2])],
        test=data.loc[data["code_presentation"].eq(PRESENTATIONS[3])],
        repeat_counts=repeat_counts,
    )
    for part in (split.train, split.validation, split.test):
        if part[TARGET].nunique() != 2:
            raise ValueError("Each chronological split needs both withdrawal outcomes.")
    return split


def expected_hits_at_k(y_true, scores, k=OUTREACH_CAPACITY):
    """Average the number of hits across all orderings of a tied boundary score."""
    targets = np.asarray(y_true)
    scores = np.asarray(scores)
    if not 0 < k <= len(scores):
        raise ValueError("k must be between 1 and the number of scored learners.")
    cutoff = np.sort(scores)[-k]
    higher = scores > cutoff
    tied = scores == cutoff
    remaining = k - higher.sum()
    return float(targets[higher].sum() + remaining * targets[tied].mean())


def ranking_metrics(y_true, scores, probabilities=False):
    k = min(OUTREACH_CAPACITY, len(y_true))
    hits = expected_hits_at_k(y_true, scores, k)
    metrics = {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "expected_hits_at_10": hits,
        "precision_at_10": hits / k,
        "recall_at_10": hits / int(np.sum(y_true)),
    }
    if probabilities:
        metrics["brier_score"] = float(brier_score_loss(y_true, scores))
    return metrics


def baseline_scores(features):
    return {
        "No activity in week two": features["active_days_week_2"].eq(0).astype("int8"),
        "Days since last activity": features["days_since_last_activity"],
    }


def candidates():
    return {
        "Logistic regression": (
            make_pipeline(
                FunctionTransformer(np.log1p),
                StandardScaler(),
                LogisticRegression(max_iter=500),
            ),
            {"logisticregression__C": [0.03, 0.3, 3.0, 30.0]},
        ),
        "Random forest": (
            RandomForestClassifier(n_estimators=180, random_state=SEED, n_jobs=2),
            {"max_depth": [3, 6, None], "min_samples_leaf": [8, 25]},
        ),
        "Gradient boosting": (
            HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
            {
                "max_iter": [60, 120],
                "max_leaf_nodes": [7, 15],
                "l2_regularization": [0.3, 3.0],
            },
        ),
    }


def selection_key(trial):
    """Higher validation average precision wins; a lower Brier score breaks ties."""
    return trial["average_precision"], -trial["brier_score"]


def tune_models(split):
    """Select hyperparameters using only training outcomes and the validation intake."""
    training_features = split.train[FEATURES]
    training_target = split.train[TARGET]
    validation_features = split.validation[FEATURES]
    validation_target = split.validation[TARGET]
    trials = []
    best_models = {}

    for name, (prototype, grid) in candidates().items():
        best_trial = None
        for params in ParameterGrid(grid):
            estimator = clone(prototype).set_params(**params)
            with threadpool_limits(limits=2):
                estimator.fit(training_features, training_target)
                scores = estimator.predict_proba(validation_features)[:, 1]
            metrics = ranking_metrics(validation_target, scores, probabilities=True)
            trial = {"model": name, "parameters": params, **metrics}
            trials.append(trial)
            key = selection_key(trial)
            if best_trial is None or key > best_trial:
                best_trial = key
                best_models[name] = (estimator, trial)

    table = pd.DataFrame(trials).sort_values(
        ["average_precision", "brier_score"],
        ascending=[False, True],
        ignore_index=True,
    )
    return table, best_models


def evaluate_models(split, best_models):
    """Refit selected configurations before scoring the untouched final intake."""
    fit = pd.concat([split.train, split.validation], ignore_index=True)
    holdout_features = split.test[FEATURES]
    holdout_target = split.test[TARGET]
    selected_name = max(
        best_models, key=lambda name: selection_key(best_models[name][1])
    )

    holdout_scores = baseline_scores(holdout_features)
    report = {
        name: ranking_metrics(holdout_target, scores)
        for name, scores in holdout_scores.items()
    }
    final_model = None
    for name, (estimator, _) in best_models.items():
        estimator = clone(estimator)
        with threadpool_limits(limits=2):
            estimator.fit(fit[FEATURES], fit[TARGET])
            scores = estimator.predict_proba(holdout_features)[:, 1]
        holdout_scores[name] = scores
        report[name] = ranking_metrics(holdout_target, scores, probabilities=True)
        if name == selected_name:
            final_model = estimator

    return selected_name, final_model, pd.DataFrame(report).T, holdout_scores
