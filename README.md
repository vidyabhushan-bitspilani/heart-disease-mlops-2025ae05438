# Heart Disease MLOps Pipeline

## 1. Project Overview

This project implements an end-to-end machine learning operations (MLOps) pipeline using the UCI Cleveland Heart Disease dataset. It covers data acquisition, exploratory data analysis (EDA), feature engineering, model development, MLflow experiment tracking, automated testing, CI/CD, Docker packaging, Kubernetes deployment, and API monitoring.

The model is intended for educational demonstration only. It is not a medical diagnostic tool.

## 2. Project Objectives

- Acquire and inspect the UCI Cleveland Heart Disease dataset.
- Perform exploratory data analysis and document data quality.
- Build a reproducible machine learning pipeline.
- Compare Logistic Regression and Random Forest models.
- Track experiments using MLflow.
- Package the trained model behind a FastAPI REST API.
- Validate functionality using automated tests.
- Automate testing and Docker image publishing with GitHub Actions.
- Deploy the API to Kubernetes.
- Implement request logging and Prometheus-compatible metrics.

## 3. Dataset Acquisition

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

Dataset source:

https://archive.ics.uci.edu/dataset/45/heart+disease

The file used by this project is `processed.cleveland.data`, containing 303 records and 13 input features, along with the original disease target.

The target is converted to a binary classification label:

- `0`: No disease indicated by the original target.
- `1`: Disease indicated by the original target (`num > 0`).

The dataset is included in the repository at:

`data/processed.cleveland.data`

To reproduce the project, clone the repository and verify that the dataset file exists before running the pipeline.

## 4. Repository Structure

```text
heart-disease-mlops-2025ae05438/
├── .github/
│   └── workflows/
│       └── ci.yaml
├── data/
│   └── processed.cleveland.data
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── models/
│   ├── heart_model.joblib
│   └── feature_schema.txt
├── reports/
│   ├── eda_summary.csv
│   ├── missing_values.csv
│   ├── class_distribution.csv
│   ├── target_distribution.png
│   ├── age_distribution.png
│   └── model_comparison.csv
├── src/
│   ├── eda.py
│   ├── train.py
│   └── api.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Environment Setup

Requirements:

- Python 3.12
- Git
- Docker for container build and execution
- `kubectl` and Minikube for local Kubernetes deployment

Clone the repository:

```bash
git clone https://github.com/vidyabhushan-bitspilani/heart-disease-mlops-2025ae05438.git
cd heart-disease-mlops-2025ae05438
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm that the dataset is available:

```bash
ls -l data/processed.cleveland.data
```

If the dataset is not present, download the `processed.cleveland.data` file from the UCI dataset source above and place it in the `data/` directory.

## 6. Exploratory Data Analysis

Run the EDA script:

```bash
python src/eda.py
```

The analysis produces summary reports and visualizations under `reports/`.

Generated outputs include:

- `eda_summary.csv`: descriptive statistics.
- `missing_values.csv`: missing-value summary.
- `class_distribution.csv`: target class counts.
- `target_distribution.png`: target distribution visualization.
- `age_distribution.png`: age distribution visualization.

The original dataset uses `?` to represent missing values in some fields. These values must be interpreted as missing data rather than ordinary numeric or categorical values.

## 7. Feature Engineering and Model Development

The prediction pipeline uses the following numeric features:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

Numeric features are processed using median imputation and standard scaling.

The categorical features are:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

Categorical features are processed using most-frequent-value imputation and one-hot encoding.

The project compares two classifiers:

1. Logistic Regression
2. Random Forest

The data is split into training and testing subsets using an 80:20 ratio, a fixed random seed of 42, and stratification on the target variable.

Run model training:

```bash
python src/train.py
```

Training produces the serialized selected pipeline at `models/heart_model.joblib`, the feature schema at `models/feature_schema.txt`, and a model comparison report at `reports/model_comparison.csv`.

The training script compares the models using held-out ROC-AUC and selects the model with the higher score. Because the held-out test set is used for model selection, the reported test metrics may be optimistic. A more rigorous future implementation would use cross-validation or a validation set for model selection and reserve the test set for final evaluation.

## 8. MLflow Experiment Tracking

MLflow is used to record model training experiments.

The training script logs model parameters, evaluation metrics, and model artifacts. The local tracking backend uses SQLite, with the database file named `mlflow.db`.

Run training to create or update the experiment records:

```bash
python src/train.py
```

Launch the MLflow interface:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open the MLflow interface in a browser at:

`http://127.0.0.1:5000`

The tracking database is a generated local artifact and is excluded from Git using `.gitignore`. Run the training script to recreate tracking records in a new environment.

## 9. Model Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

The training script writes the comparison results to `reports/model_comparison.csv`.

Previously observed results from one training run were:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8852 | 0.9016 |
| Precision | 0.8387 | 0.8438 |
| Recall | 0.9286 | 0.9643 |
| F1 score | 0.8814 | 0.9000 |
| ROC-AUC | 0.9665 | 0.9502 |

These values are run-specific and should be verified against the current generated comparison report. Logistic Regression was selected by the script because it achieved the higher ROC-AUC in this run, while Random Forest achieved higher accuracy, recall, and F1 score.

These results are not evidence of clinical validity. The dataset is small, and additional external validation would be necessary before any real-world medical use.

## 10. Automated Testing

Automated tests are located in `tests/test_api.py`.

Run the tests:

```bash
python -m pytest -q
```

The tests exercise the API behavior, including health checks and prediction requests.

The latest recorded local test run completed with four passing tests. A Starlette deprecation warning was also reported; it did not cause the tests to fail.

Tests should be rerun after changes to the API, model, dependencies, or deployment configuration.

## 11. Continuous Integration and Continuous Deployment

GitHub Actions is configured in:

`.github/workflows/ci.yaml`

The workflow runs on pushes to the main branch and includes automated checks and Docker image publishing.

The CI stage installs dependencies, verifies that the dataset is available, runs model training, and executes the automated tests.

After the CI stage succeeds on the main branch, the publish stage builds the Docker image and pushes it to Docker Hub.

Docker Hub image:

`vidyabhushanbitspilani/heart-disease-api-mlops`

Published tags include `latest`, `1.0`, and a commit-specific SHA tag.

The workflow requires the following GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

These secrets must be configured in the repository's GitHub Actions secrets settings. They must never be committed to source control.

Repository:

https://github.com/vidyabhushan-bitspilani/heart-disease-mlops-2025ae05438

## 12. Docker Packaging

The Dockerfile uses Python 3.12 slim as its base image, installs the pinned Python dependencies, copies the application source and trained model, and starts the FastAPI service using Uvicorn.

Build the image locally:

```bash
docker build -t heart-disease-api:local .
```

Run the container:

```bash
docker run --rm -p 8000:8000 heart-disease-api:local
```

Check the health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

The Docker image contains the trained model artifact required by the API. The MLflow tracking database and test files are not included in the production image.

## 13. REST API

The application is implemented with FastAPI in `src/api.py`.

### Health endpoint

Request:

```bash
curl http://127.0.0.1:8000/health
```

Expected response structure:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Prediction endpoint

The prediction endpoint accepts the 13 model input features as JSON.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "trestbps": 145,
    "chol": 233,
    "thalach": 150,
    "oldpeak": 2.3,
    "sex": 1,
    "cp": 1,
    "fbs": 1,
    "restecg": 2,
    "exang": 0,
    "slope": 3,
    "ca": 0,
    "thal": 6
  }'
```

The response includes the predicted binary class, the model's estimated disease probability, the model artifact filename, and an educational-use disclaimer.

The probability is a model output and should not be interpreted as a clinically validated estimate of an individual's medical risk.

### Prometheus metrics endpoint

Request:

```bash
curl http://127.0.0.1:8000/metrics
```

The endpoint exposes Prometheus-compatible metrics, including HTTP request counts, request duration, API error counts, and prediction counts.

## 14. Kubernetes Deployment

The project includes Kubernetes manifests under `k8s/`:

- `deployment.yaml`: application Deployment with two replicas, resource requests and limits, and health probes.
- `service.yaml`: NodePort Service.
- `ingress.yaml`: Ingress routing HTTP traffic to the application Service.

The deployment uses the published Docker Hub image:

`vidyabhushanbitspilani/heart-disease-api-mlops:latest`

### Start Minikube

If Minikube and the required tools are installed and accessible:

```bash
minikube start --driver=docker --cpus=2 --memory=4096
```

Enable the ingress controller:

```bash
minikube addons enable ingress
```

### Apply the Kubernetes manifests

```bash
kubectl apply -f k8s/
```

Check deployment status:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
kubectl get ingress
```

Wait for the deployment rollout:

```bash
kubectl rollout status deployment/heart-disease-api
```

### Access the API

The Ingress is configured to route requests to the API Service. In the recorded Minikube environment, the Ingress address was `192.168.49.2`.

Check the health endpoint:

```bash
curl http://192.168.49.2/health
```

Open the API documentation:

`http://192.168.49.2/docs`

Submit a prediction request to:

`http://192.168.49.2/predict`

The Minikube IP can change between environments or cluster restarts. Check the current address with:

```bash
kubectl get ingress
minikube ip
```

Use the current Ingress address rather than assuming the recorded address will always remain valid.

## 15. Monitoring and Logging

The FastAPI application includes request logging and Prometheus-compatible metrics.

The application records HTTP request information, including the method, endpoint, response status, and request duration.

The metrics endpoint exposes:

- `heart_api_http_requests_total`: total HTTP requests, grouped by method, endpoint, and status.
- `heart_api_request_duration_seconds`: request-duration histogram.
- `heart_api_errors_total`: API error counter.
- `heart_api_predictions_total`: prediction counter grouped by predicted class.

Inspect the metrics endpoint:

```bash
curl -s http://127.0.0.1:8000/metrics
```

For the Kubernetes deployment, use the Ingress address:

```bash
curl -s http://192.168.49.2/metrics
```

The metrics endpoint has been verified in the local Kubernetes environment. This project exposes Prometheus-formatted metrics but does not, by itself, deploy a Prometheus server or Grafana dashboard. Metrics from separate replicas are maintained by their respective application processes unless a monitoring system scrapes and aggregates them.

## 16. Reproducibility

Reproducibility is supported through:

- Pinned Python package versions in `requirements.txt`.
- A fixed random seed in the train/test split.
- A documented feature preprocessing pipeline.
- A serialized scikit-learn pipeline.
- MLflow experiment tracking.
- Automated API tests.
- Docker image packaging.
- Kubernetes deployment manifests.
- GitHub Actions CI/CD automation.

For a fresh run, install the dependencies, verify the dataset, run EDA, train the models, execute the tests, and then build and run the Docker image.

Generated artifacts such as the MLflow SQLite database are not necessarily version-controlled. They can be recreated by rerunning the appropriate scripts.

## 17. Limitations and Future Improvements

The current implementation is an educational MLOps demonstration. Several improvements would make it more robust:

1. Use a dedicated validation set or cross-validation for model selection, keeping the test set untouched until final evaluation.
2. Add explicit data validation and schema checks before training.
3. Add model versioning and an explicit model promotion process.
4. Add integration tests for container and Kubernetes deployments.
5. Deploy a Prometheus server and Grafana dashboard for centralized monitoring.
6. Add drift detection and model performance monitoring when suitable labeled data becomes available.
7. Improve dependency vulnerability scanning and container hardening.
8. Evaluate the model on external data before considering any application beyond education.
9. Add authentication, rate limiting, and transport security before exposing the API publicly.
10. Establish a clinical review and regulatory process before any healthcare use.

## 18. Assignment Evidence and Screenshots

Capture and include screenshots of the following items in the final report:

1. Dataset file and EDA outputs.
2. Model comparison report.
3. MLflow experiment interface showing recorded runs and metrics.
4. Successful local automated test execution.
5. Successful GitHub Actions CI run.
6. Successful Docker image publishing job.
7. Docker Hub repository showing the published image tags.
8. Kubernetes Deployment and Pod status.
9. Successful API health request.
10. Successful API prediction request.
11. Prometheus metrics endpoint output.
12. API documentation page at `/docs`.

Screenshots should be genuine outputs from the actual project environment. Do not substitute illustrative screenshots for evidence of successful execution.

## 19. Repository and Access Links

GitHub repository:

https://github.com/vidyabhushan-bitspilani/heart-disease-mlops-2025ae05438

Docker Hub repository:

https://hub.docker.com/r/vidyabhushanbitspilani/heart-disease-api-mlops

The API is available at the local Minikube Ingress address while the cluster and application are running. This is a local development deployment, not a permanently hosted public cloud endpoint. Access from another machine may require additional network configuration.

## 20. Conclusion

This project demonstrates an end-to-end MLOps workflow for a binary classification task. It integrates dataset analysis, preprocessing, model comparison, MLflow tracking, automated testing, CI/CD, Docker packaging, Kubernetes deployment, and API observability.

The implementation provides a reproducible foundation for further experimentation with deployment automation and model monitoring. The model is intended strictly for educational use and is not suitable for medical diagnosis.