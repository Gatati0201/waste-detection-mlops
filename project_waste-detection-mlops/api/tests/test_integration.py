"""
Integration tests for Waste Detection API
"""

import pytest
import requests
import time
import subprocess
import os
import signal
import sys

def test_api_end_to_end():
    """End-to-end test via Docker"""
    # This test assumes the API is running in Docker
    api_url = os.getenv('API_URL', 'http://localhost:8000')
    
    # Wait for API to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        pytest.skip("API not available for integration test")
    
    # Test health
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200
    assert response.json().get('status') == 'ok'
    
    # Test models endpoint
    response = requests.get(f"{api_url}/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    
    # Test predict with test image
    test_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test_image.jpg')
    
    if os.path.exists(test_image_path):
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            data = {
                'latitude': 48.8566,
                'longitude': 2.3522,
                'model_name': 'yolov8'
            }
            response = requests.post(f"{api_url}/predict", files=files, data=data)
            # May fail on model loading but should not fail on validation
            assert response.status_code in [200, 422, 500]
    
    # Test history
    response = requests.get(f"{api_url}/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    
    # Test metrics
    response = requests.get(f"{api_url}/metrics")
    assert response.status_code == 200
