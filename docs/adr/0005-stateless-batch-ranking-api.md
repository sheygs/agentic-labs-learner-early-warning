# ADR-0005: Expose a stateless batch-ranking API with strict input validation

**Status**: Accepted
**Date**: 2026-11-14

## Context

The output a mentor needs is an ordered list for one cohort ([ADR-0001](0001-rank-for-fixed-outreach-capacity.md)). A rank has no meaning for a single learner in isolation. The service should also hold as little learner data as possible, and bad input should fail loudly instead of producing a plausible-looking score.

## Decision

Serve one synchronous endpoint, `POST /predict`, that accepts a whole cohort and returns it ranked. The service keeps no state between requests.

- **Input**: 1 to 5,000 learners, each with an opaque ID and exactly 14 daily click counts.
- **Validation**: strict integers from 0 to 1,000,000, IDs matching a fixed pattern of at most 64 characters, unique IDs within a request, and unknown fields rejected. Violations return HTTP 422.
- **Output**: every learner with a rank and a risk score, in descending score order, together with the model name, model version and horizon.
- **Tie-break**: equal scores are ordered by learner ID, so identical requests return identical rankings.
- **Health**: `GET /health` reports the loaded model and backs the container and cluster probes.

## Alternatives considered

- **Score one learner per request**: the caller would have to assemble and sort the ranking, and tie-breaking would differ between clients.
- **Return only the top ten**: capacity varies by team. Returning the full ranking lets the caller choose the cutoff.
- **Coerce or clip invalid input**: a 13-day or float-valued series would silently produce a wrong feature vector.
- **Store cohorts and scores in a database**: this adds personal-data handling that the prototype does not need.

## Consequences

**Benefits**

- No learner data is persisted, and IDs can be pseudonyms.
- The contract is small enough to verify with a standard-library smoke test, which CI runs against the built image.
- Responses are reproducible and carry the model version that produced them.

**Costs**

- There is no authentication, rate limiting or request logging. The service must stay on a private network.
- A cohort larger than 5,000 must be split by the caller, and ranks from separate requests are not comparable by rank number, only by score.
- The caller is responsible for sending only learners who are still enrolled at day 14. The API cannot check this.

## Implementation

[predict.py](../../predict.py), [scripts/smoke_test.py](../../scripts/smoke_test.py) and [examples/cohort.json](../../examples/cohort.json).
