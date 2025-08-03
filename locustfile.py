"""
Locust load testing script for Food Classification API
"""

from locust import HttpUser, task, between
import os
import random
import io
import time
import zipfile
import tempfile
from PIL import Image
import numpy as np

class FoodClassificationUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        # Create sample test images
        self.test_images = self.create_test_images()
        
        # Store some sample file data for upload testing
        self.sample_files = self.create_sample_files()
    
    def create_sample_files(self):
        """Create sample files for testing file uploads"""
        files = []
        categories = ['Bread', 'Dessert', 'Meat', 'Rice', 'Soup']
        
        for category in categories[:2]:  # Just 2 categories for testing
            for i in range(2):  # 2 images per category
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                files.append({
                    'data': img_bytes.getvalue(),
                    'category': category,
                    'filename': f'{category.lower()}_{i}.jpg'
                })
        
        return files
    
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
        """Test single image prediction with timing"""
        # Select random test image
        image_data = random.choice(self.test_images)
        
        files = {'file': ('test_image.jpg', image_data, 'image/jpeg')}
        
        start_time = time.time()
        with self.client.post("/predict", files=files, catch_response=True) as response:
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('success'):
                        # Log slow predictions for monitoring
                        if processing_time > 5000:  # > 5 seconds
                            print(f"⚠️  Slow prediction: {processing_time:.2f}ms")
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
                try:
                    result = response.json()
                    if 'prediction_insights' in result or 'dataset_overview' in result:
                        response.success()
                    else:
                        response.failure("Visualizations response missing required data")
                except Exception as e:
                    response.failure(f"Failed to parse visualizations: {e}")
            else:
                response.failure(f"Visualizations failed with status {response.status_code}")
    
    @task(1)
    def test_training_status(self):
        """Test training status endpoint"""
        with self.client.get("/training/status", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'current_model' in result:
                        response.success()
                    else:
                        response.failure("Training status response invalid")
                except Exception as e:
                    response.failure(f"Failed to parse training status: {e}")
            else:
                response.failure(f"Training status failed with status {response.status_code}")
    
    @task(1)
    def test_models_current(self):
        """Test current model info endpoint"""
        with self.client.get("/models/current", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'version' in result and 'num_classes' in result:
                        response.success()
                    else:
                        response.failure("Current model response invalid")
                except Exception as e:
                    response.failure(f"Failed to parse current model: {e}")
            elif response.status_code == 503:
                # Model not loaded is acceptable
                response.success()
            else:
                response.failure(f"Current model failed with status {response.status_code}")
    
    @task(1)
    def test_upload_training_data(self):
        """Test uploading training data (occasional)"""
        if not hasattr(self, 'sample_files') or not self.sample_files:
            return
        
        # Select a random file for upload
        file_data = random.choice(self.sample_files)
        
        files = {'files': (file_data['filename'], file_data['data'], 'image/jpeg')}
        data = {'category': file_data['category']}
        
        with self.client.post("/upload/training-data", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'message' in result and 'files' in result:
                        response.success()
                    else:
                        response.failure("Upload response invalid")
                except Exception as e:
                    response.failure(f"Failed to parse upload response: {e}")
            else:
                response.failure(f"Upload failed with status {response.status_code}")

class MLOpsPipelineUser(HttpUser):
    """User class for testing the complete MLOps pipeline"""
    wait_time = between(5, 10)  # Longer wait for pipeline operations
    
    def on_start(self):
        """Initialize for pipeline testing"""
        self.test_images = self.create_test_images(3)
        self.pipeline_id = random.randint(1000, 9999)
    
    def create_test_images(self, num_images=3):
        """Create test images"""
        images = []
        for i in range(num_images):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append(img_bytes.getvalue())
        
        return images
    
    @task(3)
    def test_predict_and_monitor(self):
        """Test prediction followed by metrics check"""
        # Make prediction
        image_data = random.choice(self.test_images)
        files = {'file': (f'pipeline_test_{self.pipeline_id}.jpg', image_data, 'image/jpeg')}
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                # Follow up with metrics check
                self.client.get("/metrics")
                response.success()
            else:
                response.failure(f"Pipeline prediction failed: {response.status_code}")
    
    @task(2)
    def test_model_management(self):
        """Test model management endpoints"""
        # Check current model
        with self.client.get("/models/current", catch_response=True) as response:
            if response.status_code in [200, 503]:  # 503 if no model loaded
                # Check available models
                self.client.get("/models/versions")
                response.success()
            else:
                response.failure(f"Model management failed: {response.status_code}")
    
    @task(1)
    def test_training_pipeline(self):
        """Test training pipeline status"""
        # Check training data structure
        self.client.get("/training/data-structure")
        
        # Check training status
        with self.client.get("/training/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Training pipeline failed: {response.status_code}")

class RetrainingTestUser(HttpUser):
    """Specialized user for testing retraining workflows"""
    wait_time = between(10, 20)  # Longer waits for training operations
    
    def on_start(self):
        """Initialize for retraining testing"""
        self.test_images = self.create_training_dataset()
        self.training_zip = self.create_training_zip()
        self.user_id = random.randint(1000, 9999)
    
    def create_training_dataset(self):
        """Create a realistic training dataset"""
        categories = ['Bread', 'Dessert', 'Meat', 'Rice', 'Soup']
        dataset = {}
        
        for category in categories:
            images = []
            # Create 5-10 images per category
            for i in range(random.randint(5, 10)):
                # Create more realistic food-like images with different characteristics
                if category == 'Bread':
                    # Brownish tones for bread
                    base_color = [139, 69, 19]  # Brown
                elif category == 'Dessert':
                    # Sweet colors for desserts
                    base_color = [255, 182, 193]  # Light pink
                elif category == 'Meat':
                    # Reddish tones for meat
                    base_color = [160, 82, 45]  # Saddle brown
                elif category == 'Rice':
                    # White/beige for rice
                    base_color = [245, 245, 220]  # Beige
                else:  # Soup
                    # Warm colors for soup
                    base_color = [255, 165, 0]  # Orange
                
                # Add some variation to the base color
                img_array = np.random.normal(base_color, 30, (224, 224, 3))
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(img_array)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=85)
                img_bytes.seek(0)
                
                images.append({
                    'data': img_bytes.getvalue(),
                    'filename': f'{category.lower()}_{i}_{self.user_id}.jpg'
                })
            
            dataset[category] = images
        
        return dataset
    
    def create_training_zip(self):
        """Create a ZIP file with training data"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for category, images in self.test_images.items():
                # Create a few images per category in the ZIP
                for i, img_data in enumerate(images[:3]):  # Limit to 3 per category
                    zip_file.writestr(
                        f"{category}/{img_data['filename']}", 
                        img_data['data']
                    )
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    @task(5)
    def test_upload_individual_files(self):
        """Test uploading individual training files"""
        # Select a random category and image
        category = random.choice(list(self.test_images.keys()))
        image_data = random.choice(self.test_images[category])
        
        files = {'files': (image_data['filename'], image_data['data'], 'image/jpeg')}
        data = {'category': category}
        
        with self.client.post("/upload/training-data", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('message') and 'uploaded' in result['message'].lower():
                        response.success()
                    else:
                        response.failure("Upload didn't confirm success")
                except Exception as e:
                    response.failure(f"Upload response parsing failed: {e}")
            else:
                response.failure(f"Individual file upload failed: {response.status_code}")
    
    @task(3)
    def test_upload_zip_file(self):
        """Test uploading ZIP archive of training data"""
        files = {'zip_file': (f'training_data_{self.user_id}.zip', self.training_zip, 'application/zip')}
        
        with self.client.post("/upload/training-data-zip", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('message') and result.get('total_files', 0) > 0:
                        print(f"✅ ZIP upload successful: {result.get('total_files')} files")
                        response.success()
                    else:
                        response.failure("ZIP upload didn't process files")
                except Exception as e:
                    response.failure(f"ZIP response parsing failed: {e}")
            else:
                response.failure(f"ZIP upload failed: {response.status_code}")
    
    @task(2)
    def test_check_training_data_structure(self):
        """Test checking current training data structure"""
        with self.client.get("/training/data-structure", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'retrain_data_structure' in result:
                        files_count = result.get('total_files', 0)
                        categories_count = result.get('total_categories', 0)
                        print(f"📊 Training data: {files_count} files, {categories_count} categories")
                        response.success()
                    else:
                        response.failure("Training data structure response invalid")
                except Exception as e:
                    response.failure(f"Data structure parsing failed: {e}")
            else:
                response.failure(f"Data structure check failed: {response.status_code}")
    
    @task(1)
    def test_trigger_retraining(self):
        """Test triggering model retraining"""
        # First check if we have enough data
        data_check = self.client.get("/training/data-structure")
        if data_check.status_code == 200:
            try:
                data_info = data_check.json()
                if data_info.get('total_files', 0) < 5:
                    print("⚠️  Skipping retraining - insufficient data")
                    return
            except:
                pass
        
        # Trigger retraining with custom parameters
        payload = {
            'epochs': random.choice([5, 10, 15]),  # Vary training intensity
            'batch_size': random.choice([16, 32])   # Vary batch size
        }
        
        with self.client.post("/retrain", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    task_id = result.get('task_id')
                    if task_id:
                        print(f"🚀 Retraining started: {task_id}")
                        # Follow up by checking status
                        self.client.get(f"/training/status/{task_id}")
                        response.success()
                    else:
                        response.failure("Retraining didn't return task ID")
                except Exception as e:
                    response.failure(f"Retraining response parsing failed: {e}")
            else:
                response.failure(f"Retraining trigger failed: {response.status_code}")
    
    @task(2)
    def test_monitor_training_status(self):
        """Test monitoring training status and history"""
        with self.client.get("/training/status", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    current_model = result.get('current_model', 'Unknown')
                    recent_jobs = result.get('recent_retraining_jobs', [])
                    print(f"🤖 Current model: {current_model}, Recent jobs: {len(recent_jobs)}")
                    response.success()
                except Exception as e:
                    response.failure(f"Training status parsing failed: {e}")
            else:
                response.failure(f"Training status check failed: {response.status_code}")

class UploadStressTestUser(HttpUser):
    """User for stress testing file uploads"""
    wait_time = between(0.5, 2.0)
    
    def on_start(self):
        """Initialize for upload stress testing"""
        self.stress_images = self.create_stress_test_images()
        self.user_id = random.randint(10000, 99999)
    
    def create_stress_test_images(self):
        """Create images optimized for stress testing"""
        images = []
        categories = ['Bread', 'Dessert', 'Meat']  # Fewer categories for focused testing
        
        for category in categories:
            for i in range(3):  # 3 images per category
                # Create smaller images for faster upload
                img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=60)  # Lower quality for speed
                img_bytes.seek(0)
                
                images.append({
                    'data': img_bytes.getvalue(),
                    'category': category,
                    'filename': f'stress_{category.lower()}_{i}_{self.user_id}.jpg'
                })
        
        return images
    
    @task(10)
    def stress_test_individual_upload(self):
        """Stress test individual file uploads"""
        image_data = random.choice(self.stress_images)
        
        files = {'files': (image_data['filename'], image_data['data'], 'image/jpeg')}
        data = {'category': image_data['category']}
        
        with self.client.post("/upload/training-data", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stress upload failed: {response.status_code}")
    
    @task(3)
    def stress_test_batch_upload(self):
        """Stress test batch uploads"""
        # Upload 2-4 files at once
        batch_size = random.randint(2, 4)
        selected_images = random.sample(self.stress_images, batch_size)
        
        files = []
        for img_data in selected_images:
            files.append(('files', (img_data['filename'], img_data['data'], 'image/jpeg')))
        
        # Use the same category for simplicity
        data = {'category': selected_images[0]['category']}
        
        with self.client.post("/upload/training-data", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stress batch upload failed: {response.status_code}")

class EndToEndMLOpsUser(HttpUser):
    """Complete end-to-end MLOps workflow testing"""
    wait_time = between(15, 30)  # Longer waits for complete workflows
    
    def on_start(self):
        """Initialize for end-to-end testing"""
        self.workflow_id = random.randint(100000, 999999)
        self.test_data = self.prepare_workflow_data()
        self.workflow_state = "initialized"
    
    def prepare_workflow_data(self):
        """Prepare data for complete workflow"""
        categories = ['Bread', 'Dessert', 'Meat', 'Rice']
        data = {}
        
        for category in categories:
            images = []
            for i in range(8):  # More images for realistic training
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                
                images.append({
                    'data': img_bytes.getvalue(),
                    'filename': f'e2e_{category.lower()}_{i}_{self.workflow_id}.jpg'
                })
            
            data[category] = images
        
        return data
    
    @task(1)
    def complete_mlops_workflow(self):
        """Execute complete MLOps workflow: Upload -> Train -> Validate -> Deploy"""
        print(f"🔄 Starting E2E workflow {self.workflow_id}")
        
        # Step 1: Upload training data
        upload_success = self.upload_training_data()
        if not upload_success:
            return
        
        # Step 2: Check data structure
        data_ready = self.verify_data_structure()
        if not data_ready:
            return
        
        # Step 3: Trigger retraining
        training_task = self.start_retraining()
        if not training_task:
            return
        
        # Step 4: Monitor training progress
        self.monitor_training_progress(training_task)
        
        # Step 5: Test new model (if training completed)
        self.test_model_after_training()
        
        print(f"✅ E2E workflow {self.workflow_id} completed")
    
    def upload_training_data(self):
        """Upload training data for the workflow using ZIP"""
        print(f"📤 Uploading training data via ZIP for workflow {self.workflow_id}")
        
        # Create ZIP file with all training data
        zip_buffer = io.BytesIO()
        total_files = 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for category, images in self.test_data.items():
                # Add 3-4 images per category to ZIP
                for img_data in images[:4]:
                    zip_file.writestr(
                        f"{category}/{img_data['filename']}", 
                        img_data['data']
                    )
                    total_files += 1
        
        zip_buffer.seek(0)
        zip_filename = f'e2e_training_data_{self.workflow_id}.zip'
        
        # Upload the ZIP file
        files = {'zip_file': (zip_filename, zip_buffer.getvalue(), 'application/zip')}
        
        response = self.client.post("/upload/training-data-zip", files=files)
        if response.status_code == 200:
            try:
                result = response.json()
                uploaded_count = result.get('total_files', 0)
                print(f"✅ ZIP upload successful: {uploaded_count} files uploaded")
                return uploaded_count > 0
            except Exception as e:
                print(f"❌ ZIP upload response parsing failed: {e}")
                return False
        else:
            print(f"❌ ZIP upload failed: {response.status_code}")
            return False
    
    def verify_data_structure(self):
        """Verify uploaded data structure"""
        response = self.client.get("/training/data-structure")
        if response.status_code == 200:
            try:
                data = response.json()
                total_files = data.get('total_files', 0)
                ready_for_training = data.get('ready_for_training', False)
                print(f"📊 Data verification: {total_files} files, ready: {ready_for_training}")
                return ready_for_training
            except:
                return False
        return False
    
    def start_retraining(self):
        """Start the retraining process"""
        payload = {
            'epochs': 5,  # Quick training for testing
            'batch_size': 32
        }
        
        response = self.client.post("/retrain", json=payload)
        if response.status_code == 200:
            try:
                result = response.json()
                task_id = result.get('task_id')
                print(f"🚀 Training started: {task_id}")
                return task_id
            except:
                return None
        
        print(f"❌ Retraining failed: {response.status_code}")
        return None
    
    def test_model_after_training(self):
        """Test model after potential retraining"""
        # Create a test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': (f'e2e_test_{self.workflow_id}.jpg', img_bytes.getvalue(), 'image/jpeg')}
        
        response = self.client.post("/predict", files=files)
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('success'):
                    print(f"🎯 Post-training prediction successful")
                    return True
            except:
                pass
        
        print(f"❌ Post-training prediction failed")
        return False

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

# MLOps pipeline testing
locust -f locustfile.py --host=http://localhost:8000 MLOpsPipelineUser

# ===== NEW: RETRAINING & UPLOAD TESTING =====

# Test retraining workflows (uploads + training)
locust -f locustfile.py --host=http://localhost:8000 RetrainingTestUser --users 5 --spawn-rate 1 --run-time 600s

# Test upload stress (high volume file uploads)
locust -f locustfile.py --host=http://localhost:8000 UploadStressTestUser --users 10 --spawn-rate 2 --run-time 300s

# End-to-end MLOps workflow testing (complete pipeline)
locust -f locustfile.py --host=http://localhost:8000 EndToEndMLOpsUser --users 2 --spawn-rate 1 --run-time 900s

# Mixed workload (all user types together)
locust -f locustfile.py --host=http://localhost:8000 --users 20 --spawn-rate 2 --run-time 600s

# ===== PRODUCTION TESTING =====

# Headless mode with specific parameters
locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 5 --run-time 300s --headless

# Step load test
locust -f locustfile.py --host=http://localhost:8000 StepLoadShape

# Production-like testing
locust -f locustfile.py --host=https://food-classifier-api.redbeach-cd51ecd1.southafricanorth.azurecontainerapps.io

# Quick health check test
locust -f locustfile.py --host=http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s --headless

# Stress test for deployment validation
locust -f locustfile.py --host=http://localhost:8000 StressTestUser --users 100 --spawn-rate 10 --run-time 120s

# Performance monitoring (long-running test)
locust -f locustfile.py --host=http://localhost:8000 --users 20 --spawn-rate 2 --run-time 1800s --headless --html report.html

# ===== SPECIALIZED SCENARIOS =====

# Test only retraining (no predictions)
locust -f locustfile.py --host=http://localhost:8000 RetrainingTestUser --users 3 --spawn-rate 1 --run-time 1200s --headless

# Test only uploads (validate upload capacity)
locust -f locustfile.py --host=http://localhost:8000 UploadStressTestUser --users 15 --spawn-rate 3 --run-time 300s --headless

# Complete pipeline validation (before production deployment)
locust -f locustfile.py --host=http://localhost:8000 EndToEndMLOpsUser --users 1 --spawn-rate 1 --run-time 1800s --headless --html pipeline_report.html
"""
