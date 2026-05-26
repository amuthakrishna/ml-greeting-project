from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "krishnamoorthy",
    "start_date": datetime(2026, 5, 26),
}

with DAG(
    dag_id="ml_etl_pipeline",
    default_args=default_args,
    schedule="@daily",
    catchup=False,
) as dag:

    # Task 1 - Data Ingestion
    data_ingestion = BashOperator(
        task_id="data_ingestion",
        bash_command="python /home/krishnamoorthy/Documents/udemy_learning/26052026/ml-greeting-project/src/data_ingestion.py"
    )

    # Task 2 - Data Preparation
    data_preparation = BashOperator(
        task_id="data_preparation",
        bash_command="python /home/krishnamoorthy/Documents/udemy_learning/26052026/ml-greeting-project/src/data_preparation.py"
    )

    # Task 3 - Model Training
    model_training = BashOperator(
        task_id="model_training",
        bash_command="python /home/krishnamoorthy/Documents/udemy_learning/26052026/ml-greeting-project/src/train_model.py"
    )

    # Pipeline Order
    data_ingestion >> data_preparation >> model_training