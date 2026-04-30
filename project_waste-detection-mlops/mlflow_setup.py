"""
MLflow Setup Script - Register models in MLflow registry
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Model names to register
MODEL_NAMES = [
    'waste-detector-yolov8',
    'waste-detector-yolo26',
    'waste-detector-rtdetr',
    'waste-detector-rtdetrv2',
    'waste-detector-rfdetr',
    'waste-detector-dfine',
    'waste-detector-deim-dfine',
    'waste-detector-fusion-model',
]

def create_mock_model():
    """Create a simple mock model for registration"""
    class MockModel:
        def predict(self, data):
            return [{'class': 'rubbish', 'confidence': 0.75}]
    return MockModel()

def register_models():
    """Register all models in MLflow"""
    client = MlflowClient()
    
    for model_name in MODEL_NAMES:
        try:
            # Check if model already exists
            try:
                versions = client.get_latest_versions(model_name, stages=["Production"])
                if versions:
                    print(f"Model {model_name} already registered (v{versions[0].version})")
                    continue
            except:
                pass
            
            # Create experiment
            experiment_name = f"{model_name}-exp"
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id
            
            # Log model
            with mlflow.start_run(experiment_id=experiment_id):
                mock_model = create_mock_model()
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=mock_model,
                    registered_model_name=model_name
                )
                print(f"Registered model: {model_name}")
            
            # Transition to Production
            versions = client.get_latest_versions(model_name)
            if versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=versions[0].version,
                    stage="Production"
                )
                print(f"  -> Transitioned to Production (v{versions[0].version})")
                
        except Exception as e:
            print(f"Error registering {model_name}: {e}")

if __name__ == "__main__":
    print("Setting up MLflow models...")
    register_models()
    print("MLflow setup complete!")
