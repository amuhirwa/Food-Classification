"""
Locust load testing script for Food Classification API
"""

from locust import HttpUser, task, between
import os
import random
import io
from PIL import Image
import numpy as np

class FoodClassificationUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        # Create sample test images
        self.test_images = self.create_test_images()
    
    def create_test_images(self, num_images=5):
        """Create sample test images for testing"""
        images = []
        
        for i in range(num_images):
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append(img_bytes.getvalue())
        
        return images
    
    @task(10)
    def test_health_check(self):
        """Test health endpoint - highest weight for monitoring"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(8)
    def test_single_prediction(self):
        """Test single image prediction"""
        # Select random test image
        image_data = random.choice(self.test_images)
        
        files = {'file': ('test_image.jpg', image_data, 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        response.success()
                    else:
                        response.failure("Prediction was not successful")
                except Exception as e:
                    response.failure(f"Failed to parse response: {e}")
            else:
                response.failure(f"Prediction failed with status {response.status_code}")
    
    @task(3)
    def test_batch_prediction(self):
        """Test batch image prediction"""
        # Create batch of 2-5 images
        batch_size = random.randint(2, min(5, len(self.test_images)))
        
        files = []
        for i in range(batch_size):
            image_data = random.choice(self.test_images)
            files.append(('files', (f'test_image_{i}.jpg', image_data, 'image/jpeg')))
        
        with self.client.post("/predict/batch", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'batch_results' in result:
                        response.success()
                    else:
                        response.failure("Batch prediction response invalid")
                except Exception as e:
                    response.failure(f"Failed to parse batch response: {e}")
            else:
                response.failure(f"Batch prediction failed with status {response.status_code}")
    
    @task(2)
    def test_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'total_predictions' in result:
                        response.success()
                    else:
                        response.failure("Metrics response invalid")
                except Exception as e:
                    response.failure(f"Failed to parse metrics: {e}")
            else:
                response.failure(f"Metrics failed with status {response.status_code}")
    
    @task(2)
    def test_model_info(self):
        """Test model info endpoint"""
        with self.client.get("/model-info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Model info failed with status {response.status_code}")
    
    @task(1)
    def test_visualizations(self):
        """Test visualizations endpoint"""
        with self.client.get("/visualizations", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Visualizations failed with status {response.status_code}")

class FoodClassificationUserHeavyLoad(HttpUser):
    """User class for heavy load testing"""
    wait_time = between(0.1, 0.5)  # Faster requests
    
    def on_start(self):
        """Called when a user starts"""
        self.test_images = self.create_test_images(3)  # Fewer images for speed
    
    def create_test_images(self, num_images=3):
        """Create smaller test images for faster upload"""
        images = []
        
        for i in range(num_images):
            # Create smaller images for faster upload
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=70)  # Lower quality for speed
            img_bytes.seek(0)
            
            images.append(img_bytes.getvalue())
        
        return images
    
    @task(15)
    def test_single_prediction_fast(self):
        """Fast single predictions for stress testing"""
        image_data = random.choice(self.test_images)
        files = {'file': ('test.jpg', image_data, 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Fast prediction failed: {response.status_code}")
    
    @task(5)
    def test_health_fast(self):
        """Fast health checks"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

# Load testing scenarios
class StressTestUser(HttpUser):
    """User for stress testing with high frequency requests"""
    wait_time = between(0.01, 0.1)
    
    @task
    def stress_test_health(self):
        """Stress test health endpoint"""
        self.client.get("/health")

# Custom load test shapes (optional)
from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    A step load shape that increases users in steps
    """
    
    step_time = 30  # seconds
    step_load = 10  # users per step
    spawn_rate = 2  # users per second
    time_limit = 300  # total test time
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None
        
        current_step = run_time // self.step_time
        return (current_step * self.step_load, self.spawn_rate)

class SpikeLoadShape(LoadTestShape):
    """
    A spike load shape for testing sudden load increases
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 2)  # Normal load
        elif run_time < 120:
            return (50, 10)  # Spike
        elif run_time < 180:
            return (10, 2)  # Back to normal
        else:
            return None

# Example usage commands:
"""
# Basic load test
locust -f locustfile.py --host=http://localhost:8000

# Heavy load test with specific user class
locust -f locustfile.py --host=http://localhost:8000 FoodClassificationUserHeavyLoad

# Headless mode with specific parameters
locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 300s --headless

# Step load test
locust -f locustfile.py --host=http://localhost:8000 StepLoadShape

# Test specific scenarios
locust -f locustfile.py --host=http://localhost:8000 --tags fast
locust -f locustfile.py --host=http://localhost:8000 --tags batch
"""
