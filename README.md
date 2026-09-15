# Titanic Survival MLOps Pipeline

An end-to-end MLOps project that trains a model to predict passenger survival on the Titanic. The
pipeline prepares the data, trains and evaluates a PyTorch model, records metrics in MLflow, and
deploys a FastAPI service with a Streamlit web interface.

## Quick start

### Requirements

- Git
- Docker Desktop or Docker Engine
- Docker Compose v2 (`docker compose`)
- Approximately 5 GB of free disk space for the initial build

Python does not need to be installed on the host machine. The dataset is included in the repository,
and Docker installs all required dependencies automatically.

### Run the complete pipeline

```bash
git clone https://github.com/Yameteshka/Titanic-Survival-Lab.git
cd Titanic-Survival-Lab
docker compose -p titanic-airflow -f services/airflow/docker-compose.yml up -d --build
```

The first build may take several minutes. After the containers start, Airflow automatically runs the
complete pipeline. The prediction services become available after the first successful DAG run.

| Service | Address | Credentials |
|---|---|---|
| Web application | <http://localhost:8501> | — |
| FastAPI documentation | <http://localhost:8000/docs> | — |
| Airflow | <http://localhost:8080> | `admin` / `admin` |
| MLflow | <http://localhost:5000> | — |

Check the container status:

```bash
docker compose -p titanic-airflow -f services/airflow/docker-compose.yml ps
docker compose -p titanic-mlops -f code/deployment/docker-compose.yml ps
```

Stop all services:

```bash
docker compose -p titanic-airflow -f services/airflow/docker-compose.yml down
docker compose -p titanic-mlops -f code/deployment/docker-compose.yml down
```

## Architecture

The project consists of one scheduled pipeline and two prediction services:

```text
                         Airflow (every 5 minutes)
                                    │
                                    ▼
Raw Titanic CSV ──► Clean and split data ──► Train model ──► Evaluate model
                                                                    │
                                                                    ├──► MLflow metrics
                                                                    │
                                                                    ▼
                                                             Saved artifacts
                                                                    │
                                                                    ▼
User ──► Streamlit web app ──HTTP──► FastAPI ──► PyTorch model ──► Prediction
```

Airflow executes the stages in order. If an earlier stage fails, later stages are not deployed.

### 1. Data engineering

Input: `data/raw/titanic.csv`

The data stage:

1. validates the required columns and allowed values;
2. fills missing numerical values with medians;
3. fills missing categorical values with the most frequent value;
4. removes duplicate rows and `Age`/`Fare` outliers using the 1.5×IQR rule;
5. creates a reproducible stratified 80/20 train-test split.

Outputs:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/cleaning_report.json`

### 2. Model engineering

The training stage creates several derived features, including family size, travelling alone,
fare per person, logarithmic fare, and age group. Numerical features are standardized, categorical
features are one-hot encoded, and the preprocessing pipeline is fitted only on the training data.

The model is a deterministic PyTorch binary classifier with one linear output neuron. It is evaluated
on the held-out test set using accuracy, precision, recall, F1, ROC-AUC, log loss, and a confusion
matrix. MLflow records the parameters, metrics, and generated artifacts.

Outputs:

- `models/model.pt`
- `models/preprocessor.joblib`
- `models/model_metadata.json`
- `models/metrics.json`

### 3. Deployment

FastAPI loads the packaged model and exposes prediction endpoints. Streamlit provides the user
interface and communicates with FastAPI over HTTP; it does not load the model directly.

The API and web application have separate Dockerfiles, images, containers, ports, and health checks.
After every successful training run, Airflow rebuilds both images, starts the services, and performs
an API smoke test.

### Automation

The Airflow DAG is named `titanic_survival_ml_pipeline` and runs every five minutes:

```text
validate raw data
    → clean and split data
    → feature engineering and training
    → evaluation and MLflow logging
    → build and start prediction services
    → API smoke test
```

- Schedule: `*/5 * * * *`
- Catchup: disabled
- Maximum active runs: 1

## Dataset

The repository includes the public Titanic training dataset with 891 passenger records. Its source
and feature description are documented in [`data/README.md`](data/README.md).

`Survived` is the binary target: `0` means that the passenger did not survive and `1` means that the
passenger survived. The model uses `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
Passenger names, tickets, cabins, and identifiers are excluded from the model inputs.

## Repository structure

```text
.
├── code/
│   ├── datasets/                 # validation, cleaning, and splitting
│   ├── models/                   # features, training, evaluation, inference
│   └── deployment/
│       ├── api/                  # FastAPI service and Dockerfile
│       ├── app/                  # Streamlit application and Dockerfile
│       └── docker-compose.yml
├── data/
│   ├── raw/titanic.csv
│   └── processed/
├── models/                       # packaged model and evaluation metrics
├── mlruns/                       # local MLflow tracking data
├── scripts/run_pipeline.py       # local pipeline entry point
├── services/airflow/             # Airflow image, DAG, and Compose stack
├── tests/
└── requirements.txt
```

## Web application

The application supports:

- prediction for a single passenger using form fields;
- example passenger profiles;
- batch prediction from an uploaded CSV file;
- downloading batch results;
- viewing the deployed model version and test metrics.

Required CSV columns:

```csv
Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
1,female,35,1,0,75.0,C
3,male,28,0,0,8.05,S
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service and model status |
| `GET` | `/model-info` | Model metadata and test metrics |
| `POST` | `/predict` | Predict one passenger |
| `POST` | `/predict-batch` | Predict JSON records or an uploaded CSV file |
| `GET` | `/docs` | Interactive API documentation |

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass":1,"sex":"female","age":35,"sibsp":1,"parch":0,"fare":75,"embarked":"C"}'
```

## Local development without Airflow

This section is optional. The complete project can be run using Docker only, as described in
[Quick start](#quick-start).

Create a Python 3.11 or 3.12 virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell with a standard Python installation:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux, macOS, or MSYS2:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the data and model stages:

```bash
python -m scripts.run_pipeline
```

Run the test suite:

```bash
python -m pytest -q
```

Start only the prediction services:

```bash
docker compose -p titanic-mlops -f code/deployment/docker-compose.yml up -d --build
```

## Verification

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8501/_stcore/health
```

Inspect recent Airflow runs:

```bash
docker compose -p titanic-airflow -f services/airflow/docker-compose.yml exec \
  airflow-scheduler airflow dags list-runs \
  -d titanic_survival_ml_pipeline --no-backfill
```

Check DAG import errors:

```bash
docker compose -p titanic-airflow -f services/airflow/docker-compose.yml exec \
  airflow-scheduler airflow dags list-import-errors
```

## Troubleshooting

| Problem | Solution |
|---|---|
| Docker cannot connect to the engine | Start Docker Desktop or Docker Engine and wait until it is ready. |
| The web application is not available immediately | Wait for the first Airflow DAG run to finish successfully. |
| Port 8000, 8501, 8080, or 5000 is busy | Stop the process or Compose stack currently using the port. |
| The API health check fails | Check that the four files under `models/` exist and rebuild the deployment stack. |
| The DAG is not visible | Wait for the scheduler to parse it, then check DAG import errors. |
| Airflow cannot start the prediction containers | Ensure the Docker socket is available to the Airflow containers. |
