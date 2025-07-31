"""
Deployment script for food classification model
"""

import os
import sys
import subprocess
import argparse
import json
import time
import requests
from datetime import datetime

def check_requirements():
    """Check if all requirements are met for deployment"""
    print("🔍 Checking deployment requirements...")
    
    # Check if model exists
    model_path = "../models/food_classifier_final.h5"
    if not os.path.exists(model_path):
        print("❌ Model file not found. Please train a model first.")
        return False
    
    # Check if Docker is available
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available or not installed")
        return False
    
    # Check if requirements.txt exists
    if not os.path.exists("../requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    print("✅ All requirements met")
    return True

def build_docker_image(tag="food-classifier:latest"):
    """Build Docker image"""
    print(f"🐳 Building Docker image: {tag}")
    
    try:
        # Change to project directory
        os.chdir("..")
        
        # Build Docker image
        result = subprocess.run(
            ["docker", "build", "-t", tag, "."],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Docker image built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build Docker image: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_docker_container(tag="food-classifier:latest", port=8000):
    """Run Docker container"""
    print(f"🚀 Starting Docker container on port {port}")
    
    try:
        # Stop existing container if running
        subprocess.run(
            ["docker", "stop", "food-classifier-app"],
            capture_output=True
        )
        subprocess.run(
            ["docker", "rm", "food-classifier-app"],
            capture_output=True
        )
        
        # Run new container
        cmd = [
            "docker", "run", "-d",
            "--name", "food-classifier-app",
            "-p", f"{port}:8000",
            "-v", f"{os.path.abspath('models')}:/app/models:ro",
            "-v", f"{os.path.abspath('uploads')}:/app/uploads",
            tag
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        container_id = result.stdout.strip()
        
        print(f"✅ Container started successfully: {container_id[:12]}")
        return container_id
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Docker container: {e}")
        print(f"Error output: {e.stderr}")
        return None

def wait_for_service(url, timeout=120):
    """Wait for service to be ready"""
    print(f"⏳ Waiting for service to be ready at {url}")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except requests.RequestException:
            pass
        
        time.sleep(5)
        print(".", end="", flush=True)
    
    print("\n❌ Service did not start within timeout")
    return False

def test_deployment(base_url):
    """Test the deployed service"""
    print(f"🧪 Testing deployment at {base_url}")
    
    tests = [
        ("Health Check", "GET", "/health"),
        ("Model Info", "GET", "/model-info"),
        ("Metrics", "GET", "/metrics"),
    ]
    
    for test_name, method, endpoint in tests:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED (Status: {response.status_code})")
                
        except requests.RequestException as e:
            print(f"❌ {test_name}: FAILED (Error: {e})")

def deploy_to_cloud(provider, **kwargs):
    """Deploy to cloud provider"""
    print(f"☁️ Deploying to {provider}")
    
    if provider == "heroku":
        deploy_to_heroku()
    elif provider == "aws":
        deploy_to_aws(**kwargs)
    elif provider == "gcp":
        deploy_to_gcp(**kwargs)
    elif provider == "azure":
        deploy_to_azure(**kwargs)
    else:
        print(f"❌ Unsupported cloud provider: {provider}")

def deploy_to_heroku():
    """Deploy to Heroku"""
    print("🚀 Deploying to Heroku...")
    
    # Check if Heroku CLI is available
    try:
        subprocess.run(["heroku", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Heroku CLI not found. Please install Heroku CLI first.")
        return
    
    # Create Procfile if it doesn't exist
    if not os.path.exists("Procfile"):
        with open("Procfile", "w") as f:
            f.write("web: uvicorn api.main:app --host 0.0.0.0 --port $PORT\n")
    
    # Initialize git repo if needed
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"])
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", "Initial commit"])
    
    # Create Heroku app
    try:
        app_name = f"food-classifier-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        subprocess.run(["heroku", "create", app_name], check=True)
        
        # Deploy
        subprocess.run(["git", "push", "heroku", "main"], check=True)
        
        print(f"✅ Deployed to Heroku: https://{app_name}.herokuapp.com")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Heroku deployment failed: {e}")

def deploy_to_aws(**kwargs):
    """Deploy to AWS (placeholder)"""
    print("🚀 AWS deployment not implemented yet")
    print("Please use AWS CLI or AWS Console to deploy the Docker image")

def deploy_to_gcp(**kwargs):
    """Deploy to Google Cloud Platform (placeholder)"""
    print("🚀 GCP deployment not implemented yet")
    print("Use: gcloud run deploy --source .")

def deploy_to_azure(**kwargs):
    """Deploy to Azure (placeholder)"""
    print("🚀 Azure deployment not implemented yet")
    print("Use Azure CLI or Azure Portal to deploy")

def show_deployment_info(container_id, port):
    """Show deployment information"""
    print("\n" + "="*50)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print("="*50)
    print(f"📍 API URL: http://localhost:{port}")
    print(f"📊 Dashboard: http://localhost:{port}/")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"🐳 Container ID: {container_id[:12] if container_id else 'N/A'}")
    print("\n📋 Available Endpoints:")
    print("  - GET /health - Health check")
    print("  - POST /predict - Single image prediction")
    print("  - POST /predict/batch - Batch predictions")
    print("  - POST /upload-data - Upload training data")
    print("  - POST /retrain - Trigger retraining")
    print("  - GET /metrics - System metrics")
    print("  - GET /visualizations - Data insights")
    print("\n🛠️ Management Commands:")
    print(f"  - View logs: docker logs food-classifier-app")
    print(f"  - Stop service: docker stop food-classifier-app")
    print(f"  - Remove container: docker rm food-classifier-app")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Deploy food classification model')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the service')
    parser.add_argument('--tag', default='food-classifier:latest', help='Docker image tag')
    parser.add_argument('--cloud', choices=['heroku', 'aws', 'gcp', 'azure'], help='Deploy to cloud provider')
    parser.add_argument('--local-only', action='store_true', help='Only deploy locally')
    parser.add_argument('--build-only', action='store_true', help='Only build Docker image')
    parser.add_argument('--test-only', action='store_true', help='Only test existing deployment')
    
    args = parser.parse_args()
    
    base_url = f"http://localhost:{args.port}"
    
    # Test existing deployment
    if args.test_only:
        test_deployment(base_url)
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Build Docker image
    if not build_docker_image(args.tag):
        return
    
    if args.build_only:
        print("✅ Docker image built successfully")
        return
    
    # Run locally
    container_id = run_docker_container(args.tag, args.port)
    if not container_id:
        return
    
    # Wait for service to be ready
    if not wait_for_service(base_url):
        return
    
    # Test deployment
    test_deployment(base_url)
    
    # Show deployment info
    show_deployment_info(container_id, args.port)
    
    # Deploy to cloud if requested
    if args.cloud and not args.local_only:
        deploy_to_cloud(args.cloud)

if __name__ == "__main__":
    main()
