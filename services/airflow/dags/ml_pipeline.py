from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

PROJECT_ROOT = "/opt/project"
COMPOSE_FILE = f"{PROJECT_ROOT}/code/deployment/docker-compose.yml"
DEFAULT_ARGS = {"owner": "Julia", "retries": 1, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="titanic_survival_ml_pipeline",
    description="Data engineering, model engineering, and deployment for Titanic survival",
    schedule="*/5 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["PMLDL", "MLOps", "Titanic"],
) as dag:
    with TaskGroup("data_engineering") as data_engineering:
        validate_raw_data = BashOperator(
            task_id="validate_raw_data",
            bash_command=f"cd {PROJECT_ROOT} && python -m code.datasets.validate",
        )
        clean_and_split_data = BashOperator(
            task_id="clean_and_split_data",
            bash_command=f"cd {PROJECT_ROOT} && python -m code.datasets.preprocess",
        )
        validate_raw_data >> clean_and_split_data

    with TaskGroup("model_engineering") as model_engineering:
        feature_engineering_and_train = BashOperator(
            task_id="feature_engineering_and_train",
            bash_command=f"cd {PROJECT_ROOT} && python -m code.models.train",
        )
        evaluate_and_log_model = BashOperator(
            task_id="evaluate_and_log_model",
            bash_command=f"cd {PROJECT_ROOT} && python -m code.models.evaluate",
        )
        feature_engineering_and_train >> evaluate_and_log_model

    with TaskGroup("deployment") as deployment:
        build_and_start_services = BashOperator(
            task_id="build_and_start_services",
            bash_command=f"docker compose -p titanic-mlops -f {COMPOSE_FILE} up -d --build --remove-orphans",
        )
        smoke_test_api = BashOperator(
            task_id="smoke_test_api",
            bash_command=(
                "python - <<'PY'\n"
                "import json, time, urllib.request\n"
                "base='http://host.docker.internal:8000'\n"
                "for attempt in range(20):\n"
                "    try:\n"
                "        with urllib.request.urlopen(base + '/health', timeout=4) as response:\n"
                "            health=json.load(response)\n"
                "        assert health['status']=='ok' and health['model_loaded'] is True\n"
                "        break\n"
                "    except Exception:\n"
                "        if attempt == 19: raise\n"
                "        time.sleep(3)\n"
                "payload=json.dumps({'pclass':1,'sex':'female','age':35,'sibsp':1,'parch':0,'fare':75,'embarked':'C'}).encode()\n"
                "request=urllib.request.Request(base + '/predict', data=payload, headers={'Content-Type':'application/json'})\n"
                "with urllib.request.urlopen(request, timeout=10) as response:\n"
                "    result=json.load(response)\n"
                "assert result['predicted_class'] in (0,1) and 0 <= result['probability_survived'] <= 1\n"
                "print('Smoke test passed:', result)\n"
                "PY"
            ),
        )
        build_and_start_services >> smoke_test_api

    data_engineering >> model_engineering >> deployment
