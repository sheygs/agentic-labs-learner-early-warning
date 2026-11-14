"""Tune the candidate models, evaluate chronologically and export the selected pipeline."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
import sklearn

from data import (
    FEATURES,
    HORIZON_DAYS,
    MODULE,
    OBSERVATION_DAYS,
    PRESENTATIONS,
    TARGET,
    build_dataset,
)
from download_data import DATA_DIR, ensure_data
from modeling import OUTREACH_CAPACITY, evaluate_models, split_cohorts, tune_models

ARTIFACT = Path("artifacts/model.joblib")


def export_model(destination, model, metadata):
    """Replace the serving artifact only after serialization has succeeded."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        joblib.dump({"model": model, "metadata": metadata}, temp_path, compress=3)
        temp_path.chmod(0o644)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--archive", type=Path, help="Path to a previously downloaded UCI ZIP."
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--report", type=Path, default=Path("reports/metrics.json"))
    args = parser.parse_args()

    dataset, audit = build_dataset(ensure_data(args.data_dir, args.archive))
    split = split_cohorts(dataset)
    _, best_models = tune_models(split)
    winner, model, results, _ = evaluate_models(split, best_models)

    metadata = {
        "model_name": winner,
        "parameters": best_models[winner][1]["parameters"],
        "model_version": "day14-v1",
        "course": MODULE,
        "observation_days": OBSERVATION_DAYS,
        "prediction_horizon_days": HORIZON_DAYS,
        "feature_names": FEATURES,
        "scikit_learn_version": sklearn.__version__,
    }
    cohorts = audit.assign(repeated_learner_exclusions=split.repeat_counts)
    analysis_rows = pd.concat([split.train, split.validation, split.test])
    counts = analysis_rows.groupby("code_presentation")[TARGET].agg(["size", "sum"])
    cohorts["modeled_enrolments"] = counts["size"]
    cohorts["modeled_withdrawals"] = counts["sum"]
    cohorts["modeled_withdrawal_rate"] = counts["sum"] / counts["size"]
    report = {
        "model": metadata,
        "split": {
            "training_intakes": list(PRESENTATIONS[:2]),
            "validation_intake": PRESENTATIONS[2],
            "held_out_intake": PRESENTATIONS[3],
            "test_enrolments": len(split.test),
            "test_withdrawals": int(split.test[TARGET].sum()),
        },
        "selection_metric": "validation average precision",
        "outreach_capacity": OUTREACH_CAPACITY,
        "cohorts": cohorts.to_dict(orient="index"),
        "validation_best_per_model": {
            name: trial for name, (_, trial) in best_models.items()
        },
        "held_out_metrics": results.replace({np.nan: None}).to_dict(orient="index"),
    }

    export_model(args.artifact, model, metadata)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"Selected on {PRESENTATIONS[2]}: {winner} {metadata['parameters']}")
    print(
        f"Held out: {PRESENTATIONS[3]} ({len(split.test)} learners, {report['split']['test_withdrawals']} withdrawals)"
    )
    print(results.round(4).to_string())
    print(f"Saved model to {args.artifact} and metrics to {args.report}")


if __name__ == "__main__":
    main()
