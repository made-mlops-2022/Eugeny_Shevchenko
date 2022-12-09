from datetime import datetime
# import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.empty import EmptyOperator
from docker.types import Mount
from utils import LOCAL_DATA_DIR, default_args

# PROJECT_PATH = os.environ.get('PROJECT_PATH', None)
with DAG(
    'generate_data',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2022, 11, 1)
) as dag:
    start = EmptyOperator(task_id="start-generate")
    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        auto_remove=True,
        do_xcom_push=False,
        mount_tmp_dir=False,
        # !!! HOST folder(NOT IN CONTAINER) replace with yours !!!
        mounts=[Mount(source="/home/evg/Документы/ml_ops/Eugeny_Shevchenko/airflow_ml_dags/data",
                      target="/data", type='bind')]
    )
    stop = EmptyOperator(task_id="stop-generate")
    start >> download >> stop
