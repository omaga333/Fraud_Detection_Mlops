from datetime import datetime, timedelta
import os
import glob
from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.providers.standard.operators.python import PythonOperator
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'datamasterylab.com',
    'depends_on_past': False,
    'start_date': datetime(2025, 3, 3),
    'max_active_runs': 1,
}

def _validate_environment(**context):
    """Python-native environment validation"""
    env_exists = os.path.exists('/app/.env')
    config_exists = os.path.exists('/app/config.yaml')
    
    logger.info(f'.env exists: {env_exists}')
    logger.info(f'config.yaml exists: {config_exists}')
    
    if not (env_exists and config_exists):
        raise AirflowException("Environment files (.env or config.yaml) are missing in /app!")
    logger.info('Environment is valid!')

def _train_model(**context):
    """Airflow wrapper for training task"""
    from fraud_detection_training import FraudDetectionTraining
    try:
        logger.info('Initializing fraud detection training')
        trainer = FraudDetectionTraining()
        model, precision = trainer.train_model()

        return { 'status': 'success', 'precision': precision }
    except Exception as e:
        logger.error('Training failed: %s', str(e), exc_info=True)
        raise AirflowException(f'Model training failed: {str(e)}')

def _cleanup_resources(**context):
    """Python-native cleanup task"""
    files = glob.glob('/app/tmp/*.pkl')
    for f in files:
        try:
            os.remove(f)
            logger.info(f"Removed temporary file: {f}")
        except Exception as e:
            logger.warning(f"Could not remove {f}: {e}")

with DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Fraud detection model training pipeline',
    schedule='0 3 * * *',
    catchup=False,
    tags=['fraud', 'ML']
) as dag:
    
    validate_environment = PythonOperator(
        task_id='validate_environment',
        python_callable=_validate_environment,
        queue='default'
    )

    training_task = PythonOperator(
        task_id='execute_training',
        python_callable=_train_model,
        queue='default'
    )

    cleanup_task = PythonOperator(
        task_id='cleanup_resources',
        python_callable=_cleanup_resources,
        trigger_rule='all_done',
        queue='default'
    )

    validate_environment >> training_task >> cleanup_task

    dag.doc_md = """
    ## Fraud Detection Training Pipeline
    
    Daily training of fraud detection model using:
    - Transaction data from Kafka
    - XGBoost classifier with precision optimisation
    - MLFlow for experiment tracking
    """