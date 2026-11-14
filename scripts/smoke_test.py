"""Check the published API contract using only the Python standard library."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def post(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--save", type=Path, help="Save the verified API response.")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    sample_path = Path(__file__).resolve().parents[1] / "examples" / "cohort.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    with urlopen(f"{url}/health", timeout=10) as response:
        health = json.load(response)
    assert health["status"] == "ok"

    result = post(f"{url}/predict", sample)
    ranked = result["ranked_learners"]
    assert health["model_version"] == result["model_version"]
    assert result["horizon_days"] == 30
    assert {item["learner_id"] for item in ranked} == {
        row["learner_id"] for row in sample["learners"]
    }
    assert [item["rank"] for item in ranked] == list(range(1, len(ranked) + 1))
    scores = [item["risk_score"] for item in ranked]
    assert all(0 <= score <= 1 for score in scores)
    assert scores == sorted(scores, reverse=True)

    for invalid in (
        {"learners": [{"learner_id": "invalid", "daily_clicks": [0] * 13}]},
        {"learners": [{"learner_id": "invalid", "daily_clicks": [-1] + [0] * 13}]},
        {"learners": [sample["learners"][0], sample["learners"][0]]},
    ):
        try:
            post(f"{url}/predict", invalid)
        except HTTPError as error:
            assert error.code == 422
        else:
            raise AssertionError("Invalid input should return HTTP 422.")

    if args.save:
        args.save.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Passed health, ranking and request validation at {url}.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
