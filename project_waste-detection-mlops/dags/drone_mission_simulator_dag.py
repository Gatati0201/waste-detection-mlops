"""
Drone Mission Simulator DAG
Simulates drone returning from patrol missions
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import subprocess
import sys
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def run_mission_simulator():
    """Execute the drone mission simulator script"""
    script_path = '/opt/airflow/generate_patrol_db.py'
    if os.path.exists(script_path):
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise Exception(f"Mission simulator failed: {result.stderr}")
        return result.stdout
    else:
        raise FileNotFoundError(f"Script not found: {script_path}")

with DAG(
    'drone_mission_simulator',
    default_args=default_args,
    description='Simulate drone patrol missions',
    schedule_interval='*/5 * * * *',  # Every 5 minutes for testing
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['drone', 'simulation'],
) as dag:
    
    simulate_mission = PythonOperator(
        task_id='simulate_mission',
        python_callable=run_mission_simulator,
    )
    
    # Trigger the sync DAG after mission simulation
    trigger_sync = TriggerDagRunOperator(
        task_id='trigger_drone_patrol_sync',
        trigger_dag_id='drone_patrol_sync',
        wait_for_completion=False,
    )
    
    simulate_mission >> trigger_sync
