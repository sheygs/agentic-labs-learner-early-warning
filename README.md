# Learner Early Warning

**Machine Learning Zoomcamp midterm · Inspired by Agentic Labs Academy**

A mentor can only check in with a handful of learners each week. This project ranks a course intake at the end of its first two weeks so a mentor can decide who to contact first. The model estimates the risk of withdrawing in the next 30 days using daily course activity.

The prototype uses public Open University data from a historical university course. The demo scores serve as a ranking. Using the scores for Academy outreach would require validation on Academy cohorts and a separate evaluation of whether check-ins improve retention.

## Results

On the held-out **2014J** intake, **30 of 1,726** eligible learners withdrew within the scoring horizon. Logistic regression was selected on the previous intake, 2014B, and refitted before measuring test performance.

| Method | Average precision | ROC AUC | Expected withdrawals among 10 check-ins |
| --- | ---: | ---: | ---: |
| Random shortlist | — | — | 0.17 |
| No activity in week two | 0.0244 | 0.6098 | 0.32 |
| Days since last activity | 0.0372 | 0.6808 | 0.55 |
| **Logistic regression (served model)** | **0.0426** | **0.7105** | **0.55** |
| Random forest | 0.0449 | 0.7567 | 0.55 |
| Histogram gradient boosting | 0.0382 | 0.6646 | 0.50 |

Average precision evaluates the entire ranking. At an outreach capacity of ten, logistic regression matched the recency rule and returned fewer than one expected hit. A tie at the cutoff is averaged over the learners with equal scores, so the expected count can be fractional.

Random forest had the highest test average precision. The exported model stays the validation winner to keep the test intake independent of the selection. The full intake counts, parameters and metrics are in [metrics.json](reports/metrics.json), and the figures and feature importance are in [notebook.ipynb](notebook.ipynb).

## Data and evaluation

[Download the public Open University Learning Analytics Dataset from UCI](https://archive.ics.uci.edu/dataset/349/open%2Buniversity%2Blearning%2Banalytics%2Bdataset). The download is about 45 MB. The script verifies its SHA-256 hash and extracts the required CSV files (about 450 MB). The source dataset was created by Jakub Kuzilek, Martin Hlosta and Zdenek Zdrahal and is distributed under CC BY 4.0. [Dataset DOI](https://doi.org/10.24432/C5KK69) · [Data descriptor](https://doi.org/10.1038/sdata.2017.171)

| Choice | Implementation |
| --- | --- |
| Population | Learners in course BBB, registered by day 13 and still enrolled when day 14 starts. One record per learner, using their earliest eligible enrolment. |
| Observation | Days 0–13 relative to the course start. Aggregate the click log into 14 daily counts. |
| Features | Active days and total clicks in each of the first two weeks, plus days since last observed activity. Days without a recorded click count as zero. |
| Target | Withdrawal on days 14–43, inclusive. Records marked as withdrawn with an unknown withdrawal date are excluded so unknown outcomes stay out of the negative class. |
| Split | Train on 2013B and 2013J. Tune and select on 2014B. Refit using those three intakes and evaluate on 2014J. |
| Selection | Compare 18 configurations across logistic regression, random forest and histogram gradient boosting. Select by validation average precision and break exact ties with the Brier score. |

The activity transformation is shared by training and inference. The predictors exclude identity, final results, grades and demographic attributes. Learner IDs are used to join records and remove repeat enrolments, while final results help exclude withdrawals with missing dates. The first two weeks of course clicks are a proxy for engagement. They miss offline study and may reflect logging gaps. Results come from a single course with 30 positive cases in the final intake. Learners with identical scores are ordered by learner ID in the API to keep responses reproducible.

## Run the project

Use **Python 3.12** and run these commands from the repository root in macOS or Linux. You can open the executed notebook directly on GitHub. To rerun it, the optional notebook requirements are below.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m unittest discover -s tests -v
```

If you already downloaded the original UCI ZIP, run `python download_data.py --archive /path/to/oulad.zip`. The data downloader skips extraction when all required files are present. The training script saves `artifacts/model.joblib` and `reports/metrics.json`. A small trained model is included so you can also try inference before downloading the data. Load only a model artifact from this project or one you trained locally.

To rerun the notebook, install its dependencies, open JupyterLab, and use **Run All Cells**.

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebook.ipynb
```

### Serve with Docker

The included model lets you run this demo using Docker alone.

```bash
docker build -t learner-early-warning:1.0 .
docker run --rm --name learner-early-warning -p 127.0.0.1:8000:8000 learner-early-warning:1.0
```

Open [API docs](http://localhost:8000/docs), or use a second terminal to send a saved request and verify the response.

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/cohort.json
python scripts/smoke_test.py
```

For each still-enrolled learner, supply an opaque ID and exactly 14 non-negative daily click counts in course-day order. The service returns learners in descending risk-score order so a mentor can take the first ten. [Sample response](reports/api-demo.json) · [Verified Docker run](reports/docker-deployment.png)

You can also serve the model directly from the virtual environment.

```bash
python -m uvicorn predict:app --host 127.0.0.1 --port 8000
```

### Kubernetes deployment

The Zoomcamp rubric accepts deployment to a local Kubernetes cluster. With Docker, [kind](https://kind.sigs.k8s.io/docs/user/quick-start/) and `kubectl` installed, build the image as shown above and run the following.

```bash
kind create cluster --name learner-early-warning
kind load docker-image learner-early-warning:1.0 --name learner-early-warning
kubectl --context kind-learner-early-warning apply -f deploy/kubernetes.yaml
kubectl --context kind-learner-early-warning rollout status deployment/learner-early-warning --timeout=180s
kubectl --context kind-learner-early-warning port-forward service/learner-early-warning 8001:80
```

While the port-forward runs, open [Kubernetes service docs](http://localhost:8001/docs) or run `python scripts/smoke_test.py --url http://localhost:8001` in another terminal. The manifest uses a private ClusterIP service and health probes. To remove the local cluster, use `kind delete cluster --name learner-early-warning`.

The manifest was applied on a local K3s cluster. [Video of the captured rollout and HTTP test](reports/kubernetes-test.mp4) · [Deployment transcript](reports/deployment-transcript.txt) · [Rollout summary](reports/kubernetes-deployment.png)

## Submit for peer review

Create a public GitHub repository and push the extracted project folder. Run `git rev-parse HEAD` after pushing and submit the repository URL and that commit on the [Zoomcamp project page](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm). Check the cohort page for peer-review dates.

## Rubric evidence

| Midterm criterion | Where to review |
| --- | --- |
| Problem description | Mentor workflow, scoring window and target in this README. |
| Exploratory data analysis | [Notebook](notebook.ipynb) sections 1, 2 and 5 show input checks, cohort prevalence, activity distributions and permutation importance. |
| Model training and tuning | Notebook sections 3 and 4 compare three model families, parameter grids, two decision rules and an untouched test intake. |
| Export training to a script | [train.py](train.py) reproduces the model artifact and evaluation using the same code as the notebook. |
| Reproducibility and environment | Public dataset with download script, pinned Python dependencies and virtual-environment instructions. |
| Model deployment and containerisation | [predict.py](predict.py), [Dockerfile](Dockerfile) and the HTTP smoke test above. |
| Cloud or Kubernetes bonus | [Deployment and Service manifest](deploy/kubernetes.yaml) with local cluster instructions and [a video of the successful rollout and prediction request](reports/kubernetes-test.mp4). |

[Official project rubric](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/projects)

This project was designed and implemented with AI assistance. The README records the decisions a reviewer can reproduce, including the label cutoff, validation split and metric used for model selection. [Zoomcamp AI usage policy](https://datatalksclub.github.io/docs/courses/zoomcamp-logistics/ai-usage/)
