FROM docker:27-cli AS docker_cli
FROM apache/airflow:2.10.5-python3.11
USER root
COPY --from=docker_cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker_cli /usr/local/libexec/docker/cli-plugins/docker-compose /usr/local/libexec/docker/cli-plugins/docker-compose
USER airflow
COPY services/airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
    && pip install --no-cache-dir -r /tmp/requirements-airflow.txt
