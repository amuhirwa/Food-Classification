"""
FastAPI application for food classification MLOps pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import shutil
import json
from datetime import datetime
from typing import List, Optional
import pandas as pd
from PIL import Image
import zipfile
import logging

# Import custom modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.prediction import FoodClassificationPredictor
    from src.model import FoodClassificationModel
    from src.preprocessing import FoodDataPreprocessor
except ImportError:
    # Fallback import
    from prediction import FoodClassificationPredictor
    from model import FoodClassificationModel
    from preprocessing import FoodDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Classification MLOps API",
    description="API for food image classification with MLOps capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_DIR = "../models"
DATA_DIR = "../data"
UPLOAD_DIR = "../uploads"
RETRAIN_DATA_DIR = "../retrain_data"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RETRAIN_DATA_DIR, exist_ok=True)

# Initialize predictor
try:
    model_path = os.path.join(MODEL_DIR, "food_classifier_final.h5")
    if os.path.exists(model_path):
        predictor = FoodClassificationPredictor(
            model_path=model_path,
            class_mappings_dir=MODEL_DIR
        )
        logger.info("Predictor initialized successfully")
    else:
        predictor = None
        logger.warning("Model not found. Train a model first.")
except Exception as e:
    predictor = None
    logger.error(f"Error initializing predictor: {str(e)}")

# Store for background tasks and metrics
task_status = {}
prediction_logs = []
model_metrics = {
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "uptime_start": datetime.now(),
    "last_prediction": None
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint with API documentation
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Classification MLOps API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            .endpoint { background-color: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #2196F3; }
        </style>
    </head>
    <body>
        <h1>Food Classification MLOps API</h1>
        <p>Welcome to the Food Classification API with MLOps capabilities.</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> /health - Health check
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /predict - Predict single image
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /predict/batch - Predict multiple images
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /upload-data - Upload data for retraining
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /retrain - Trigger model retraining
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /metrics - Get model metrics and uptime
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /model-info - Get model information
        </div>
        
        <p><a href="/docs">Interactive API Documentation</a></p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    model_status = "available" if predictor and predictor.model else "unavailable"
    uptime = datetime.now() - model_metrics["uptime_start"]
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "uptime_seconds": uptime.total_seconds(),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    """
    Predict food class for a single uploaded image
    """
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        image_io = io.BytesIO(image_bytes)
        
        # Make prediction
        result = predictor.predict_from_bytes(image_io, return_probabilities=True)
        
        if result:
            # Update metrics
            model_metrics["total_predictions"] += 1
            model_metrics["successful_predictions"] += 1
            model_metrics["last_prediction"] = datetime.now().isoformat()
            
            # Log prediction
            log_entry = {
                "filename": file.filename,
                "prediction": result["predicted_class"],
                "confidence": result["confidence"],
                "timestamp": result["timestamp"]
            }
            prediction_logs.append(log_entry)
            
            # Keep only last 1000 logs
            if len(prediction_logs) > 1000:
                prediction_logs.pop(0)
            
            return {
                "success": True,
                "filename": file.filename,
                "prediction": result
            }
        else:
            model_metrics["total_predictions"] += 1
            model_metrics["failed_predictions"] += 1
            raise HTTPException(status_code=500, detail="Prediction failed")
            
    except Exception as e:
        model_metrics["total_predictions"] += 1
        model_metrics["failed_predictions"] += 1
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Predict food classes for multiple uploaded images
    """
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Read image bytes
            image_bytes = await file.read()
            image_io = io.BytesIO(image_bytes)
            
            # Make prediction
            result = predictor.predict_from_bytes(image_io)
            
            if result:
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "prediction": result
                })
                model_metrics["successful_predictions"] += 1
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Prediction failed"
                })
                model_metrics["failed_predictions"] += 1
            
            model_metrics["total_predictions"] += 1
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            model_metrics["failed_predictions"] += 1
            model_metrics["total_predictions"] += 1
    
    model_metrics["last_prediction"] = datetime.now().isoformat()
    
    return {
        "batch_results": results,
        "total_processed": len(files),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]])
    }

@app.post("/upload-data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload data for model retraining (ZIP file with organized folders)
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    try:
        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, f"retrain_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
        
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract ZIP file
        extract_path = upload_path.replace('.zip', '_extracted')
        
        with zipfile.ZipFile(upload_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Analyze uploaded data
        data_info = analyze_uploaded_data(extract_path)
        
        return {
            "success": True,
            "message": "Data uploaded successfully",
            "upload_path": upload_path,
            "extract_path": extract_path,
            "data_info": data_info
        }
        
    except Exception as e:
        logger.error(f"Data upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

def analyze_uploaded_data(data_path):
    """
    Analyze uploaded training data
    """
    analysis = {
        "total_files": 0,
        "classes": {},
        "valid_images": 0,
        "invalid_files": 0
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for root, dirs, files in os.walk(data_path):
        class_name = os.path.basename(root)
        
        if class_name not in analysis["classes"]:
            analysis["classes"][class_name] = 0
        
        for file in files:
            analysis["total_files"] += 1
            
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                analysis["valid_images"] += 1
                analysis["classes"][class_name] += 1
            else:
                analysis["invalid_files"] += 1
    
    return analysis

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, data_path: Optional[str] = None):
    """
    Trigger model retraining with new data
    """
    if not data_path:
        # Use the most recent uploaded data
        upload_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('_extracted')]
        if not upload_files:
            raise HTTPException(status_code=400, detail="No training data available")
        
        data_path = os.path.join(UPLOAD_DIR, max(upload_files))
    
    # Generate task ID
    task_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    task_status[task_id] = {
        "status": "started",
        "started_at": datetime.now().isoformat(),
        "data_path": data_path
    }
    
    # Start background retraining task
    background_tasks.add_task(retrain_model, task_id, data_path)
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Retraining started in background"
    }

async def retrain_model(task_id: str, data_path: str):
    """
    Background task for model retraining
    """
    try:
        task_status[task_id]["status"] = "preprocessing"
        
        # Initialize preprocessor and model
        preprocessor = FoodDataPreprocessor()
        
        # Load class mappings
        preprocessor.load_class_mappings(MODEL_DIR)
        
        # Create data generators
        train_gen, val_gen = preprocessor.create_data_generators(data_path)
        
        task_status[task_id]["status"] = "training"
        
        # Load existing model for retraining
        existing_model_path = os.path.join(MODEL_DIR, "food_classifier_final.h5")
        
        food_model = FoodClassificationModel(
            num_classes=len(preprocessor.classes),
            model_dir=MODEL_DIR
        )
        
        # Retrain model
        history = food_model.retrain_model(
            train_gen, val_gen, 
            base_model_path=existing_model_path if os.path.exists(existing_model_path) else None,
            epochs=20
        )
        
        if history:
            task_status[task_id]["status"] = "completed"
            task_status[task_id]["completed_at"] = datetime.now().isoformat()
            
            # Update global predictor with new model
            global predictor
            new_model_path = max([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) 
                                if f.startswith("food_classifier_retrained")], 
                               key=os.path.getctime)
            
            predictor = FoodClassificationPredictor(
                model_path=new_model_path,
                class_mappings_dir=MODEL_DIR
            )
            
            logger.info(f"Model retrained successfully: {new_model_path}")
        else:
            task_status[task_id]["status"] = "failed"
            task_status[task_id]["error"] = "Training failed"
            
    except Exception as e:
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["error"] = str(e)
        logger.error(f"Retraining error: {str(e)}")

@app.get("/retrain/status/{task_id}")
async def get_retrain_status(task_id: str):
    """
    Get status of retraining task
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status[task_id]

@app.get("/metrics")
async def get_metrics():
    """
    Get model metrics and system status
    """
    uptime = datetime.now() - model_metrics["uptime_start"]
    
    metrics = {
        **model_metrics,
        "uptime_hours": uptime.total_seconds() / 3600,
        "success_rate": (model_metrics["successful_predictions"] / 
                        max(model_metrics["total_predictions"], 1)) * 100,
        "recent_predictions": prediction_logs[-10:] if prediction_logs else []
    }
    
    return metrics

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the current model
    """
    if not predictor or not predictor.model:
        return {"error": "Model not available"}
    
    return predictor.get_model_info()

@app.get("/visualizations")
async def get_visualizations():
    """
    Get data visualizations and insights
    """
    try:
        # Analyze prediction logs for insights
        if not prediction_logs:
            return {"message": "No predictions made yet"}
        
        df = pd.DataFrame(prediction_logs)
        
        # Class distribution
        class_counts = df['prediction'].value_counts().to_dict()
        
        # Confidence distribution
        confidence_stats = {
            "mean": df['confidence'].mean(),
            "std": df['confidence'].std(),
            "min": df['confidence'].min(),
            "max": df['confidence'].max()
        }
        
        # Predictions over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        hourly_predictions = df.groupby(df['timestamp'].dt.hour).size().to_dict()
        
        visualizations = {
            "class_distribution": class_counts,
            "confidence_statistics": confidence_stats,
            "predictions_by_hour": hourly_predictions,
            "total_predictions": len(prediction_logs)
        }
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return {"error": f"Failed to generate visualizations: {str(e)}"}

@app.get("/download-logs")
async def download_logs():
    """
    Download prediction logs as CSV
    """
    if not prediction_logs:
        raise HTTPException(status_code=404, detail="No logs available")
    
    df = pd.DataFrame(prediction_logs)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return JSONResponse(
        content={"csv_data": csv_buffer.getvalue()},
        headers={"Content-Disposition": "attachment; filename=prediction_logs.csv"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
