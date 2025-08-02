"""
FastAPI application for food classification MLOps pipeline
"""

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import os
import io
import shutil
import json
import uuid
from datetime import datetime
from typing import List, Optional
import pandas as pd
from PIL import Image
import zipfile
import logging
import time

# Import custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.prediction import FoodClassificationPredictor
    from src.model import FoodClassificationModel
    from src.preprocessing import FoodDataPreprocessor
except ImportError:
    # Fallback import
    from prediction import FoodClassificationPredictor
    from model import FoodClassificationModel
    from preprocessing import FoodDataPreprocessor

# Import API modules
from database import get_db, DatasetCRUD, ModelCRUD, PredictionCRUD, RetrainingCRUD, get_system_stats
from schemas import (
    DatasetUploadResponse, ModelVersionResponse, PredictionResponse, 
    RetrainingTaskResponse, HealthResponse, ModelInfo, SystemStats,
    AnalyticsResponse, BatchUploadResponse, ErrorResponse, SystemMetricsResponse
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Classification MLOps API",
    description="API for food image classification with MLOps capabilities using SQLAlchemy",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
os.makedirs(MODEL_DIR, exist_ok=True)

# Model metrics tracking
model_metrics = {
    "uptime_start": datetime.now(),
    "total_predictions": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_uploads": 0,
    "total_retraining_jobs": 0,
    "average_processing_time": 0.0,
    "model_loads": 0,
    "errors": []
}

# Recent predictions log (keep last 100)
prediction_logs = []

# Initialize predictor (will be loaded when model is available)
predictor = None
model_trainer = None  # Will be initialized when needed
preprocessor = FoodDataPreprocessor()

def load_predictor(model_version=None):
    """Load the predictor with the specified model version or latest"""
    global predictor, model_metrics
    try:
        # Determine model path
        if model_version is None or model_version == 'latest':
            model_path = os.path.join(MODEL_DIR, "food_classifier_latest.h5")
            if not os.path.exists(model_path):
                # Fallback to the original model
                model_path = os.path.join(MODEL_DIR, "food_classifier_final.h5")
        else:
            model_path = os.path.join(MODEL_DIR, f"food_classifier_v{model_version}.h5")
        
        if os.path.exists(model_path):
            predictor = FoodClassificationPredictor(
                model_path=model_path,
                class_mappings_dir=MODEL_DIR
            )
            model_metrics["model_loads"] += 1
            
            # Get model version info
            if 'latest' in model_path:
                version_info = "latest"
            elif '_v' in model_path:
                version_info = model_path.split('_v')[-1].split('.')[0]
            else:
                version_info = "original"
            
            logger.info(f"Predictor loaded successfully from {model_path} (version: {version_info})")
            return True
        else:
            logger.warning(f"Model file not found: {model_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing predictor: {e}")
        model_metrics["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": f"Model loading failed: {str(e)}"
        })
        predictor = None
        return False

# Try to load predictor on startup
load_predictor()

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="../ui"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard"""
    try:
        with open("../ui/dashboard.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>MLOps Dashboard - Upload dashboard.html to ui/ folder</h1>")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the current model
    """
    if not predictor or not predictor.model:
        return {"error": "Model not available"}
    
    return predictor.get_model_info()


@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    # Get active model info
    active_model = ModelCRUD.get_active_model(db)
    active_model_name = f"{active_model.model_name} v{active_model.version}" if active_model else None
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=predictor is not None,
        database_ready=True,
        active_model=active_model_name
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info(db: Session = Depends(get_db)):
    """Get model information"""
    if predictor is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    
    active_model = ModelCRUD.get_active_model(db)
    
    return ModelInfo(
        model_architecture="Transfer Learning with pre-trained CNN",
        input_shape=[224, 224, 3],
        num_classes=len(predictor.class_names) if hasattr(predictor, 'class_names') else 11,
        preprocessing="Rescaling to [0,1], Resize to 224x224",
        version=f"{active_model.model_name} v{active_model.version}" if active_model else None
    )

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Predict food class for a single uploaded image
    """
    global model_metrics, prediction_logs
    start_time = time.time()
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

            # Log prediction to database
            active_model = ModelCRUD.get_active_model(db)
            if active_model:
                PredictionCRUD.create_prediction(
                    db=db,
                    model_version_id=active_model.id,
                    filename=file.filename,
                    predicted_class=result[0]['class'],
                    predicted_class_idx=result[0].get('class_index', 0),
                    confidence=result[0]['confidence'],
                    processing_time_ms= (time.time() - start_time) * 1000,
                    all_probabilities=result
                )

            
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


# @app.post("/predict", response_model=PredictionResponse)
# async def predict_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     """Predict food category from uploaded image"""
#     global model_metrics, prediction_logs
    
#     if predictor is None:
#         model_metrics["failed_predictions"] += 1
#         raise HTTPException(status_code=503, detail="Model not available. Train a model first.")
    
#     if not file.content_type.startswith('image/'):
#         model_metrics["failed_predictions"] += 1
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     start_time = time.time()
    
#     try:
#         # Read image bytes
#         image_bytes = await file.read()
#         image_io = io.BytesIO(image_bytes)
        
#         # Make prediction
#         predictions = predictor.predict_from_bytes(image_io, return_probabilities=True)
#         processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
#         # Update metrics
#         model_metrics["total_predictions"] += 1
#         model_metrics["successful_predictions"] += 1
        
#         # Update average processing time
#         current_avg = model_metrics["average_processing_time"]
#         total_preds = model_metrics["successful_predictions"]
#         model_metrics["average_processing_time"] = (
#             (current_avg * (total_preds - 1) + processing_time) / total_preds
#         )
        
#         # Log prediction
#         prediction_log = {
#             "timestamp": datetime.now().isoformat(),
#             "filename": file.filename,
#             "predicted_class": predictions[0]['class'],
#             "confidence": predictions[0]['confidence'],
#             "processing_time_ms": processing_time
#         }
#         prediction_logs.append(prediction_log)
        
#         # Keep only last 100 predictions
#         if len(prediction_logs) > 100:
#             prediction_logs.pop(0)
        
#         # Log prediction to database
#         active_model = ModelCRUD.get_active_model(db)
#         if active_model:
#             PredictionCRUD.create_prediction(
#                 db=db,
#                 model_version_id=active_model.id,
#                 filename=file.filename,
#                 predicted_class=predictions[0]['class'],
#                 predicted_class_idx=predictions[0].get('class_index', 0),
#                 confidence=predictions[0]['confidence'],
#                 processing_time_ms=processing_time,
#                 all_probabilities=predictions
#             )
        
#         # Format response
#         formatted_predictions = [
#             {
#                 "class": pred['class'],
#                 "confidence": pred['confidence'],
#                 "class_index": pred.get('class_index', i)
#             }
#             for i, pred in enumerate(predictions)
#         ]
        
#         return PredictionResponse(
#             filename=file.filename,
#             predictions=formatted_predictions,
#             processing_time_ms=processing_time,
#             timestamp=datetime.now()
#         )
    
#     except Exception as e:
#         model_metrics["failed_predictions"] += 1
#         model_metrics["errors"].append({
#             "timestamp": datetime.now().isoformat(),
#             "error": f"Prediction failed: {str(e)}"
#         })
#         logger.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch_images(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """
    Predict food classes for multiple uploaded images
    """
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not available")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    results = []
    active_model = ModelCRUD.get_active_model(db)
    start_time = time.time()
    
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
                processing_time = (time.time() - start_time) * 1000   
            # Log prediction to database
                if active_model:
                    PredictionCRUD.create_prediction(
                        db=db,
                        model_version_id=active_model.id,
                        filename=file.filename,
                        predicted_class=result[0]['class'],
                        predicted_class_idx=result[0].get('class_index', 0),
                        confidence=result[0]['confidence'],
                        processing_time_ms=processing_time,
                        all_probabilities=result
                    )

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


# @app.post("/predict/batch")
# async def predict_batch(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
#     """Predict food categories for multiple images"""
#     if predictor is None:
#         raise HTTPException(status_code=503, detail="Model not available")
    
#     results = []
#     active_model = ModelCRUD.get_active_model(db)
    
#     for file in files:
#         if not file.content_type.startswith('image/'):
#             continue
        
#         start_time = time.time()
        
#         try:
#             image_bytes = await file.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             predictions = predictor.predict_image(image)
#             processing_time = (time.time() - start_time) * 1000
            
#             # Log prediction to database
#             if active_model:
#                 PredictionCRUD.create_prediction(
#                     db=db,
#                     model_version_id=active_model.id,
#                     filename=file.filename,
#                     predicted_class=predictions[0]['class'],
#                     predicted_class_idx=predictions[0].get('class_index', 0),
#                     confidence=predictions[0]['confidence'],
#                     processing_time_ms=processing_time,
#                     all_probabilities=predictions
#                 )
            
#             formatted_predictions = [
#                 {
#                     "class": pred['class'],
#                     "confidence": pred['confidence'],
#                     "class_index": pred.get('class_index', i)
#                 }
#                 for i, pred in enumerate(predictions)
#             ]
            
#             results.append({
#                 "filename": file.filename,
#                 "predictions": formatted_predictions,
#                 "processing_time_ms": processing_time
#             })
            
#         except Exception as e:
#             results.append({
#                 "filename": file.filename,
#                 "error": str(e)
#             })
    
#     return {
#         "results": results,
#         "total_processed": len(results),
#         "timestamp": datetime.now()
#     }


@app.post("/upload/training-data", response_model=BatchUploadResponse)
async def upload_training_data(
    files: List[UploadFile] = File(...),
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Upload new training data for retraining"""
    global model_metrics
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate category name if provided
    if category:
        # Clean category name (remove special characters, normalize spaces)
        category = category.strip().replace('/', '_').replace('\\', '_')
        if not category:
            raise HTTPException(status_code=400, detail="Invalid category name")
    
    uploaded_files = []
    errors = []
    
    # Create category directory if specified
    if category:
        category_dir = os.path.join(RETRAIN_DATA_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        logger.info(f"Created/using category directory: {category_dir}")
    
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            errors.append(f"{file.filename}: Not an image file (content-type: {file.content_type})")
            continue
        
        # Validate file extension
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            errors.append(f"{file.filename}: Unsupported file extension")
            continue
        
        try:
            # Read and validate image
            content = await file.read()
            
            # Validate file size (max 10MB)
            if len(content) > 10 * 1024 * 1024:
                errors.append(f"{file.filename}: File too large (max 10MB)")
                continue
            
            # Try to open image to validate it's a valid image
            try:
                img = Image.open(io.BytesIO(content))
                img.verify()  # Verify it's a valid image
            except Exception:
                errors.append(f"{file.filename}: Invalid or corrupted image file")
                continue
            
            # Save file
            if category:
                file_path = os.path.join(RETRAIN_DATA_DIR, category, file.filename)
            else:
                # If no category, save in root and warn user
                file_path = os.path.join(RETRAIN_DATA_DIR, file.filename)
                logger.warning(f"File {file.filename} uploaded without category - consider organizing by food type")
            
            # Handle duplicate filenames
            original_path = file_path
            counter = 1
            while os.path.exists(file_path):
                name, ext = os.path.splitext(original_path)
                file_path = f"{name}_{counter}{ext}"
                counter += 1
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append({
                "filename": file.filename,
                "saved_as": os.path.basename(file_path),
                "size": len(content),
                "category": category,
                "upload_timestamp": datetime.now()
            })
            
            logger.info(f"Successfully uploaded: {file.filename} -> {file_path}")
            
        except Exception as e:
            errors.append(f"{file.filename}: Upload failed - {str(e)}")
            logger.error(f"Error uploading file {file.filename}: {e}")
    
    # Update metrics
    model_metrics["total_uploads"] += len(uploaded_files)
    
    # Prepare response message
    success_count = len(uploaded_files)
    error_count = len(errors)
    
    if success_count > 0 and error_count == 0:
        message = f"Successfully uploaded {success_count} files"
    elif success_count > 0 and error_count > 0:
        message = f"Uploaded {success_count} files with {error_count} errors"
    else:
        message = f"Failed to upload files - {error_count} errors"
    
    # Log current dataset status
    try:
        if os.path.exists(RETRAIN_DATA_DIR):
            categories = [d for d in os.listdir(RETRAIN_DATA_DIR) if os.path.isdir(os.path.join(RETRAIN_DATA_DIR, d))]
            logger.info(f"Current categories in retrain data: {categories}")
    except Exception as e:
        logger.warning(f"Could not analyze retrain directory: {e}")
    
    response = BatchUploadResponse(
        message=message,
        files=uploaded_files,
        total_files=len(uploaded_files),
        timestamp=datetime.now()
    )
    
    # Add errors to response if any (extend the response model if needed)
    if errors:
        response.errors = errors
    
    return response


@app.post("/retrain", response_model=RetrainingTaskResponse)
async def retrain_model(
    background_tasks: BackgroundTasks,
    dataset_id: Optional[int] = None,
    epochs: int = 10,
    batch_size: int = 32,
    db: Session = Depends(get_db)
):
    """Start model retraining in background"""
    global model_metrics
    
    task_id = str(uuid.uuid4())
    
    # Update metrics
    model_metrics["total_retraining_jobs"] += 1
    
    # Create retraining task record
    active_model = ModelCRUD.get_active_model(db)
    training_params = {"epochs": epochs, "batch_size": batch_size}
    
    task = RetrainingCRUD.create_retraining_task(
        db=db,
        task_id=task_id,
        dataset_id=dataset_id or 0,  # Use 0 for retrain_data directory
        base_model_id=active_model.id if active_model else None,
        training_params=training_params
    )
    
    def train_model():
        """Background task for training using FoodClassificationModel.retrain_model"""
        try:
            logger.info(f"Starting model retraining with task ID: {task_id}")
            
            # Update task status to running
            RetrainingCRUD.update_retraining_task(
                db=db,
                task_id=task_id,
                status="running"
            )
            
            # Check if retrain data directory has any data
            if not os.path.exists(RETRAIN_DATA_DIR) or not os.listdir(RETRAIN_DATA_DIR):
                raise ValueError("No training data found. Please upload training data first.")
            
            # Prepare data with proper class discovery
            preprocessor_retrain = FoodDataPreprocessor()
            train_generator, val_generator, class_names = preprocessor_retrain.prepare_generators(
                data_dir=RETRAIN_DATA_DIR,
            )
            
            if not class_names:
                raise ValueError("No classes found in training data")
            
            logger.info(f"Found {len(class_names)} classes: {class_names}")
            
            # Initialize model trainer using the proper FoodClassificationModel
            global model_trainer
            if model_trainer is None:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                try:
                    from src.model import FoodClassificationModel
                except ImportError:
                    from model import FoodClassificationModel
                model_trainer = FoodClassificationModel
            
            # Create model instance with proper number of classes
            model_instance = model_trainer(num_classes=len(class_names), model_dir=MODEL_DIR)
            
            # Get the current latest model as base for retraining
            latest_model_path = os.path.join(MODEL_DIR, "food_classifier_latest.h5")
            base_model_path = latest_model_path if os.path.exists(latest_model_path) else None
            
            # Use the retrain_model method from FoodClassificationModel
            retrain_history, model_version = model_instance.retrain_model(
                new_train_generator=train_generator,
                new_val_generator=val_generator,
                base_model_path=base_model_path,
                epochs=epochs
            )
            
            if retrain_history is None or model_version is None:
                raise ValueError("Model retraining failed")
            
            # Save class mappings for the new model
            preprocessor_retrain.save_class_mappings(MODEL_DIR)
            
            # Calculate performance metrics
            final_accuracy = float(retrain_history.history['accuracy'][-1])
            final_val_accuracy = float(retrain_history.history['val_accuracy'][-1])
            
            logger.info(f"Training completed - Accuracy: {final_accuracy:.3f}, Val Accuracy: {final_val_accuracy:.3f}")
            
            # Create new model version in database
            new_model = ModelCRUD.create_model_version(
                db=db,
                model_name=f"food_classifier_v{model_version}",
                version=str(model_version),
                model_path=os.path.join(MODEL_DIR, f"food_classifier_v{model_version}.h5"),
                base_model="MobileNetV2",
                num_classes=len(class_names),
                training_params=training_params,
                performance_metrics={
                    "final_accuracy": final_accuracy,
                    "final_val_accuracy": final_val_accuracy,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "model_version": model_version
                }
            )
            
            # Update task to completed
            RetrainingCRUD.update_retraining_task(
                db=db,
                task_id=task_id,
                status="completed",
                new_model_id=new_model.id
            )
            
            # Reload predictor with latest model
            load_predictor()
            
            logger.info(f"Model retraining completed successfully: v{model_version}")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            RetrainingCRUD.update_retraining_task(
                db=db,
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
    
    background_tasks.add_task(train_model)
    
    return RetrainingTaskResponse(
        id=task.id,
        task_id=task_id,
        dataset_id=task.dataset_id,
        status=task.status,
        started_at=task.started_at,
        training_params=training_params
    )

from fastapi import UploadFile, File, HTTPException
import zipfile
import tempfile
import shutil

@app.post("/upload/training-data-zip", response_model=BatchUploadResponse)
async def upload_training_data_zip(
    zip_file: UploadFile = File(...),
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Upload a ZIP archive of training data images for retraining"""
    global model_metrics
    
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")
    
    if category:
        category = category.strip().replace('/', '_').replace('\\', '_')
        if not category:
            raise HTTPException(status_code=400, detail="Invalid category name")
    
    uploaded_files = []
    errors = []

    # Prepare directory
    category_dir = os.path.join(RETRAIN_DATA_DIR, category) if category else RETRAIN_DATA_DIR
    os.makedirs(category_dir, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, zip_file.filename)
            with open(zip_path, "wb") as f:
                content = await zip_file.read()
                f.write(content)
            
            # Extract ZIP
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file")

            # Process extracted files
            for root, _, files in os.walk(tmpdir):
                for filename in files:
                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                    file_path = os.path.join(root, filename)

                    try:
                        with open(file_path, "rb") as f:
                            content = f.read()

                        if len(content) > 10 * 1024 * 1024:
                            continue  # Skip oversized

                        try:
                            img = Image.open(io.BytesIO(content))
                            img.verify()
                        except Exception:
                            continue  # Skip invalid image

                        # Derive relative path to get category (e.g., Bread, Dessert)
                        rel_path = os.path.relpath(file_path, tmpdir)
                        parts = Path(rel_path).parts
                        if len(parts) < 2:
                            continue  # Skip if file is at root level (no category folder)

                        category = parts[0].strip().replace('/', '_').replace('\\', '_')
                        category_dir = os.path.join(RETRAIN_DATA_DIR, category)
                        os.makedirs(category_dir, exist_ok=True)

                        save_path = os.path.join(category_dir, filename)

                        # Avoid overwriting
                        base, ext = os.path.splitext(save_path)
                        counter = 1
                        while os.path.exists(save_path):
                            save_path = f"{base}_{counter}{ext}"
                            counter += 1

                        with open(save_path, "wb") as out_file:
                            out_file.write(content)

                        uploaded_files.append({
                            "filename": filename,
                            "saved_as": os.path.basename(save_path),
                            "size": len(content),
                            "category": category,
                            "upload_timestamp": datetime.now()
                        })

                        logger.info(f"Saved {filename} to {save_path}")

                    except Exception as e:
                        logger.warning(f"Failed to process {filename}: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error during upload: {str(e)}")

    model_metrics["total_uploads"] += len(uploaded_files)

    response = BatchUploadResponse(
        message=f"Uploaded {len(uploaded_files)} files with {len(errors)} errors" if errors else f"Successfully uploaded {len(uploaded_files)} files",
        files=uploaded_files,
        total_files=len(uploaded_files),
        timestamp=datetime.now()
    )

    if errors:
        return HTTPException(
            status_code=400,
            detail={
                "message": response.message,
                "errors": errors
            }
        )

    return response


@app.get("/training/data-structure")
async def get_training_data_structure():
    """Get current training data structure and statistics"""
    try:
        structure = {}
        total_files = 0
        categories = []
        
        if os.path.exists(RETRAIN_DATA_DIR):
            for item in os.listdir(RETRAIN_DATA_DIR):
                item_path = os.path.join(RETRAIN_DATA_DIR, item)
                
                if os.path.isdir(item_path):
                    # Category directory
                    image_files = [f for f in os.listdir(item_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    file_count = len(image_files)
                    structure[item] = {
                        "file_count": file_count,
                        "files": image_files[:10]  # Show first 10 files as sample
                    }
                    total_files += file_count
                    categories.append(item)
                elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Uncategorized file
                    if "uncategorized" not in structure:
                        structure["uncategorized"] = {"file_count": 0, "files": []}
                    structure["uncategorized"]["files"].append(item)
                    structure["uncategorized"]["file_count"] += 1
                    total_files += 1
        
        # Check original data structure for reference
        original_classes = []
        if os.path.exists(DATA_DIR):
            original_classes = [d for d in os.listdir(DATA_DIR) 
                              if os.path.isdir(os.path.join(DATA_DIR, d))]
        
        return {
            "retrain_data_structure": structure,
            "total_categories": len(categories),
            "total_files": total_files,
            "categories": categories,
            "original_classes": original_classes,
            "ready_for_training": total_files > 0 and len(categories) > 0,
            "recommendations": [
                "Upload at least 10 images per category for best results",
                "Ensure images are high quality and well-lit",
                "Use descriptive category names that match food types",
                "Consider balancing the number of images across categories"
            ] if total_files > 0 else [
                "No training data uploaded yet",
                "Use POST /upload/training-data with category parameter to upload images",
                "Organize images by food category (e.g., 'Bread', 'Dessert', etc.)"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing training data structure: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing data: {str(e)}")


@app.get("/training/status")
async def training_status(db: Session = Depends(get_db)):
    """Get training status and history"""
    active_model = ModelCRUD.get_active_model(db)
    models = ModelCRUD.get_models(db, limit=10)
    retraining_tasks = RetrainingCRUD.get_retraining_tasks(db, limit=5)
    
    return {
        "current_model": f"{active_model.model_name} v{active_model.version}" if active_model else "No model loaded",
        "model_versions": [
            {
                "id": model.id,
                "name": model.model_name,
                "version": model.version,
                "created": model.created_timestamp,
                "is_active": model.is_active,
                "status": model.status
            }
            for model in models
        ],
        "recent_retraining_jobs": [
            {
                "task_id": task.task_id,
                "status": task.status,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "error_message": task.error_message
            }
            for task in retraining_tasks
        ]
    }


@app.get("/stats", response_model=SystemStats)
async def get_stats(db: Session = Depends(get_db)):
    """Get API usage statistics"""
    return get_system_stats(db)


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


# @app.get("/metrics", response_model=SystemMetricsResponse)
# async def get_metrics():
#     """
#     Get model metrics and system status
#     """
#     uptime = datetime.now() - model_metrics["uptime_start"]
    
#     # Calculate success rate
#     total_preds = max(model_metrics["total_predictions"], 1)
#     success_rate = (model_metrics["successful_predictions"] / total_preds) * 100
    
#     # Keep only recent errors (last 10)
#     recent_errors = model_metrics["errors"][-10:] if model_metrics["errors"] else []
    
#     metrics = {
#         **model_metrics,
#         "uptime_hours": round(uptime.total_seconds() / 3600, 2),
#         "uptime_formatted": str(uptime).split('.')[0],  # Remove microseconds
#         "success_rate": round(success_rate, 2),
#         "failure_rate": round(100 - success_rate, 2),
#         "recent_predictions": prediction_logs[-10:] if prediction_logs else [],
#         "recent_errors": recent_errors,
#         "predictor_loaded": predictor is not None,
#         "current_time": datetime.now().isoformat()
#     }
    
#     # Remove the full errors list to avoid large responses
#     metrics_clean = {k: v for k, v in metrics.items() if k != "errors"}
    
#     return metrics_clean


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(db: Session = Depends(get_db)):
    """Get detailed analytics"""
    prediction_stats = PredictionCRUD.get_prediction_stats(db)
    system_stats = get_system_stats(db)
    
    return AnalyticsResponse(
        prediction_stats=system_stats,
        class_distribution=prediction_stats["class_distribution"],
        hourly_predictions={},  # Could implement hourly breakdown
        model_performance={}  # Could implement model performance metrics
    )


@app.get("/predictions/history")
async def prediction_history(
    limit: int = 100,
    model_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    predictions = PredictionCRUD.get_predictions(
        db=db, 
        limit=limit, 
        model_version_id=model_id
    )
    
    return {
        "predictions": [
            {
                "id": pred.id,
                "filename": pred.filename,
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "timestamp": pred.prediction_timestamp,
                "processing_time_ms": pred.processing_time_ms
            }
            for pred in predictions
        ],
        "total": len(predictions)
    }


@app.get("/models/versions")
async def list_model_versions():
    """Get list of available model versions"""
    try:
        # Initialize model instance to use utility methods
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
        try:
            from src.model import FoodClassificationModel
        except ImportError:
            from model import FoodClassificationModel
        
        model_instance = FoodClassificationModel(num_classes=11, model_dir=MODEL_DIR)
        available_models = model_instance.get_available_models()
        
        # Add current active model info
        current_version = "unknown"
        if predictor and hasattr(predictor, 'model_path'):
            model_path = predictor.model_path
            if 'latest' in model_path:
                current_version = "latest"
            elif '_v' in model_path:
                current_version = model_path.split('_v')[-1].split('.')[0]
        
        return {
            "available_models": available_models,
            "current_active_version": current_version,
            "total_versions": len(available_models)
        }
        
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/models/switch/{version}")
async def switch_model_version(version: str):
    """Switch to a specific model version"""
    try:
        # Validate version
        if version not in ['latest'] and not version.isdigit():
            raise HTTPException(status_code=400, detail="Invalid version format")
        
        # Load the specified model version
        success = load_predictor(model_version=version)
        
        if success:
            return {
                "message": f"Successfully switched to model version {version}",
                "version": version,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found or failed to load")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model version: {e}")
        raise HTTPException(status_code=500, detail=f"Error switching model: {str(e)}")


@app.get("/models/current")
async def get_current_model_info():
    """Get information about the currently active model"""
    if not predictor:
        raise HTTPException(status_code=503, detail="No model currently loaded")
    
    try:
        # Get model path and extract version info
        model_path = getattr(predictor, 'model_path', 'Unknown')
        
        if 'latest' in model_path:
            version = "latest"
        elif '_v' in model_path:
            version = model_path.split('_v')[-1].split('.')[0]
        else:
            version = "original"
        
        # Get model file stats
        model_size_mb = 0
        if os.path.exists(model_path):
            model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)
        
        # Get class information
        num_classes = len(getattr(predictor, 'class_names', []))
        class_names = getattr(predictor, 'class_names', [])
        
        return {
            "version": version,
            "model_path": model_path,
            "model_size_mb": model_size_mb,
            "num_classes": num_classes,
            "class_names": class_names,
            "loaded_at": model_metrics.get("uptime_start", datetime.now()).isoformat(),
            "predictions_made": model_metrics.get("total_predictions", 0)
        }
        
    except Exception as e:
        logger.error(f"Error getting current model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


@app.get("/models", response_model=List[ModelVersionResponse])
async def list_models(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List all model versions"""
    models = ModelCRUD.get_models(db, skip=skip, limit=limit)
    
    result = []
    for model in models:
        try:
            training_params = json.loads(model.training_params) if model.training_params else {}
            performance_metrics = json.loads(model.performance_metrics) if model.performance_metrics else {}
        except json.JSONDecodeError:
            training_params = {}
            performance_metrics = {}
        
        result.append(ModelVersionResponse(
            id=model.id,
            model_name=model.model_name,
            version=model.version,
            created_timestamp=model.created_timestamp,
            model_path=model.model_path,
            base_model=model.base_model or "",
            num_classes=model.num_classes or 0,
            training_params=training_params,
            performance_metrics=performance_metrics,
            is_active=model.is_active,
            status=model.status
        ))
    
    return result


@app.post("/models/{model_id}/activate")
async def activate_model(model_id: int, db: Session = Depends(get_db)):
    """Activate a specific model version"""
    global predictor, model_metrics
    
    model = ModelCRUD.get_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    ModelCRUD.set_active_model(db, model_id)
    
    # Try to reload predictor with new model
    if os.path.exists(model.model_path):
        try:
            predictor = FoodClassificationPredictor(
                model_path=model.model_path,
                class_mappings_dir=MODEL_DIR
            )
            model_metrics["model_loads"] += 1
            logger.info(f"Switched to model: {model.model_name} v{model.version}")
        except Exception as e:
            model_metrics["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": f"Model activation failed: {str(e)}"
            })
            logger.error(f"Error loading model: {e}")
    
    return {"message": f"Model {model.model_name} v{model.version} activated"}


@app.delete("/data/clear")
async def clear_data():
    """Clear uploaded training data"""
    try:
        if os.path.exists(RETRAIN_DATA_DIR):
            shutil.rmtree(RETRAIN_DATA_DIR)
            os.makedirs(RETRAIN_DATA_DIR, exist_ok=True)
        
        return {"message": "Training data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear data: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
