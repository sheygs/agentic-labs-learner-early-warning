# ADR-0007: Lock the environment and run a hardened, single-replica container

**Status**: Accepted
**Date**: 2026-11-14

## Context

The project is assessed on reproducibility and deployment, and the model artifact is tied to an exact scikit-learn version ([ADR-0006](0006-model-artifact-with-metadata-checks.md)). The service handles learner activity, so it should run with as few privileges as possible even as a demo.

## Decision

Pin everything that affects results, and run the service with a minimal, locked-down footprint.

**Environment**

- Python is fixed to 3.12 and every runtime dependency is pinned in `pyproject.toml` and resolved in `uv.lock`.
- The image installs with `uv sync --locked --no-dev`. CI fails if the lockfile is out of date.
- The dataset archive is verified against a pinned SHA-256 hash before extraction.

**Image**

- `.dockerignore` excludes everything by default and allows only the two runtime modules, the lock files and the model.
- The container runs as non-root user 10001 and has a health check on `/health`.

**Kubernetes**

- One replica behind a ClusterIP Service. There is no Ingress.
- Read-only root filesystem, no privilege escalation, all capabilities dropped, the default seccomp profile and no service-account token.
- CPU and memory requests and limits, with numeric libraries capped at two threads to stay within them.
- `imagePullPolicy: Never`, because the image is loaded into a local cluster and not pushed to a registry.

**Continuous integration**

- Every pull request and push to `main` runs linting, the unit tests, and a smoke test against the built image.

## Alternatives considered

- **Unpinned or range-pinned dependencies**: a new scikit-learn release would break model loading.
- **pip with `requirements.txt`**: workable, but uv gives one lockfile for runtime, dev and notebook groups and a faster install.
- **Deploy to a managed cloud service**: a local cluster satisfies the course rubric without cost or credentials.
- **Multiple replicas with autoscaling**: the service is stateless, so it would work, but a demo with one mentor-sized cohort does not need it.

## Consequences

**Benefits**

- The same commit produces the same environment, image and model behaviour.
- A compromised process has no writable filesystem, no extra privileges and no cluster credentials.
- The API contract is exercised in CI on every change.

**Costs**

- Pinned versions need deliberate upgrades, and a scikit-learn upgrade also needs a retrain.
- The manifest is local-only: it has no registry image, Ingress, TLS, network policy or autoscaling.
- One replica means a restart is a brief outage.
- The base image is pinned by tag (`python:3.12-slim`), not by digest, so image builds are not bit-for-bit reproducible.

## Implementation

[pyproject.toml](../../pyproject.toml), `uv.lock`, [Dockerfile](../../Dockerfile), `.dockerignore`, [deploy/kubernetes.yaml](../../deploy/kubernetes.yaml), [.github/workflows/ci.yml](../../.github/workflows/ci.yml) and [download_data.py](../../download_data.py).
