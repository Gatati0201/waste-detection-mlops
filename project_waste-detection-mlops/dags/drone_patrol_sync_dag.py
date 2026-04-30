"""
Drone Patrol Sync DAG
ETL pipeline: Extract -> Transform -> Load
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sqlite3
import requests
import os

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

DRONE_DB_PATH = '/data/drone_patrol.db'
APP_DB_PATH = '/data/app_detections.db'
API_URL = 'http://api:8000'

def extract(**context):
    """Extract unprocessed drone detections"""
    conn = sqlite3.connect(DRONE_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        "SELECT * FROM drone_detections WHERE processed = 0"
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    print(f"Extracted {len(rows)} unprocessed detections")
    context['ti'].xcom_push(key='extracted_data', value=rows)
    return rows

def transform(**context):
    """Filter detections with confidence >= 0.65"""
    ti = context['ti']
    rows = ti.xcom_pull(task_ids='extract', key='extracted_data')
    
    if not rows:
        print("No data to transform")
        ti.xcom_push(key='transformed_data', value=[])
        return []
    
    # Filter by confidence
    filtered = [row for row in rows if row.get('confiance', 0) >= 0.65]
    
    print(f"Transformed: {len(filtered)} of {len(rows)} detections passed filter (>= 0.65)")
    ti.xcom_push(key='transformed_data', value=filtered)
    return filtered

def load(**context):
    """Load filtered detections into app database and mark as processed"""
    ti = context['ti']
    rows = ti.xcom_pull(task_ids='transform', key='transformed_data')
    
    if not rows:
        print("No data to load")
        return
    
    # Load into app database
    conn = sqlite3.connect(APP_DB_PATH)
    processed_ids = []
    
    for row in rows:
        try:
            conn.execute(
                """INSERT INTO app_detections (timestamp, latitude, longitude, confiance, model_name, source, drone_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    row.get('timestamp'),
                    row.get('latitude'),
                    row.get('longitude'),
                    row.get('confiance'),
                    'drone-model',  # Default model for drone detections
                    'drone_patrol',
                    row.get('drone_id')
                )
            )
            processed_ids.append(row.get('id'))
        except Exception as e:
            print(f"Error inserting row {row.get('id')}: {e}")
    
    conn.commit()
    conn.close()
    
    # Mark as processed in drone database
    if processed_ids:
        conn = sqlite3.connect(DRONE_DB_PATH)
        placeholders = ','.join('?' * len(processed_ids))
        conn.execute(
            f"UPDATE drone_detections SET processed = 1 WHERE id IN ({placeholders})",
            processed_ids
        )
        conn.commit()
        conn.close()
        print(f"Marked {len(processed_ids)} rows as processed")
    
    print(f"Loaded {len(rows)} detections into app database")

with DAG(
    'drone_patrol_sync',
    default_args=default_args,
    description='ETL pipeline for drone patrol data',
    schedule_interval='*/10 * * * *',  # Every 10 minutes (or triggered by DAG 1)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['drone', 'etl'],
) as dag:
    
    extract_task = PythonOperator(
        task_id='extract',
        python_callable=extract,
    )
    
    transform_task = PythonOperator(
        task_id='transform',
        python_callable=transform,
    )
    
    load_task = PythonOperator(
        task_id='load',
        python_callable=load,
    )
    
    extract_task >> transform_task >> load_task
