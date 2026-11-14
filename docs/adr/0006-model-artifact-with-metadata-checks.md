# ADR-0006: Ship the model as a committed, metadata-checked joblib artifact

**Status**: Accepted
**Date**: 2026-11-14

## Context

The service needs a trained model at startup. Reviewers should be able to run the service without downloading 450 MB of data and retraining. A scikit-learn model serialised with one library version is not guaranteed to load, or to behave the same, under another, and a model trained with a different feature list would still return numbers.

## Decision

Export the fitted pipeline with joblib, together with a metadata record, commit it to the repository, and verify the metadata when the service starts.

- **Contents**: `{"model": pipeline, "metadata": {...}}`. Metadata holds the model name, parameters, version, course, observation days, horizon, feature names and scikit-learn version.
- **Startup checks**: the service refuses to start if the scikit-learn version, feature names, observation window or prediction horizon differ from the running code.
- **Atomic export**: the model is written to a temporary file in the target directory and then renamed over the destination, so a failed run cannot leave a partial artifact.
- **Report first**: `reports/metrics.json` is written before the artifact is replaced, so a failure cannot pair a new model with stale metrics.
- **Location**: `artifacts/model.joblib` by default, overridable with `MODEL_PATH`.

## Alternatives considered

- **Train at image build or container start**: builds would need the dataset and would take far longer, and two builds could serve different models.
- **Keep the artifact out of Git and publish it elsewhere**: more correct for large models, but it adds a registry and a download step for a file of about 1 kB.
- **Export to ONNX or another portable format**: this removes the pickle risk and the version coupling, at the cost of a conversion step and a second runtime.
- **Load without checks**: a version or schema mismatch would surface as wrong scores, not as an error.

## Consequences

**Benefits**

- `docker build` alone produces a working service.
- A mismatched model fails at startup with a clear message.
- The artifact and its metrics report are produced by the same run.

**Costs**

- Joblib files are pickles and can execute code when loaded. Only artifacts from this repository or from a local training run should be loaded.
- A binary file is versioned in Git, and nothing enforces that it matches the committed code or metrics.
- `model_version` is the fixed string `day14-v1`. A retrained model keeps the same label unless the constant is changed by hand.
- Upgrading scikit-learn requires retraining.

## Implementation

`export_model` and `main` in [train.py](../../train.py); `lifespan` in [predict.py](../../predict.py).
