"""
Database configuration and CRUD operations
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Optional, Dict, Any
import json
from datetime import datetime
import logging

from .models import Base, UploadedDataset, ModelVersion, Prediction, RetrainingTask, SystemMetric

logger = logging.getLogger(__name__)

# Database URL - SQLite for development
DATABASE_URL = "sqlite:///./mlops.db"

# Create engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatasetCRUD:
    """CRUD operations for datasets"""
    
    @staticmethod
    def create_dataset(db: Session, filename: str, file_path: str, 
                      extracted_path: str, data_info: Dict[str, Any]) -> UploadedDataset:
        """Create new dataset record"""
        db_dataset = UploadedDataset(
            filename=filename,
            file_path=file_path,
            extracted_path=extracted_path,
            total_files=data_info.get('total_files', 0),
            valid_images=data_info.get('valid_images', 0),
            classes_info=json.dumps(data_info.get('classes', {})),
            file_metadata=json.dumps(data_info)
        )
        db.add(db_dataset)
        db.commit()
        db.refresh(db_dataset)
        return db_dataset
    
    @staticmethod
    def get_datasets(db: Session, skip: int = 0, limit: int = 50) -> List[UploadedDataset]:
        """Get all datasets"""
        return db.query(UploadedDataset).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_dataset(db: Session, dataset_id: int) -> Optional[UploadedDataset]:
        """Get dataset by ID"""
        return db.query(UploadedDataset).filter(UploadedDataset.id == dataset_id).first()
    
    @staticmethod
    def update_dataset_status(db: Session, dataset_id: int, status: str):
        """Update dataset status"""
        dataset = db.query(UploadedDataset).filter(UploadedDataset.id == dataset_id).first()
        if dataset:
            dataset.status = status
            db.commit()


class ModelCRUD:
    """CRUD operations for model versions"""
    
    @staticmethod
    def create_model_version(db: Session, model_name: str, version: str, 
                           model_path: str, base_model: str, num_classes: int,
                           training_params: Dict[str, Any], 
                           performance_metrics: Optional[Dict[str, Any]] = None,
                           training_dataset_id: Optional[int] = None) -> ModelVersion:
        """Create new model version"""
        
        # Deactivate previous active models
        db.query(ModelVersion).update({"is_active": False})
        
        db_model = ModelVersion(
            model_name=model_name,
            version=version,
            model_path=model_path,
            base_model=base_model,
            num_classes=num_classes,
            training_params=json.dumps(training_params),
            performance_metrics=json.dumps(performance_metrics) if performance_metrics else None,
            is_active=True,
            training_dataset_id=training_dataset_id,
            status="completed"
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        return db_model
    
    @staticmethod
    def get_active_model(db: Session) -> Optional[ModelVersion]:
        """Get currently active model"""
        return db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    @staticmethod
    def get_models(db: Session, skip: int = 0, limit: int = 50) -> List[ModelVersion]:
        """Get all model versions"""
        return db.query(ModelVersion).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_model(db: Session, model_id: int) -> Optional[ModelVersion]:
        """Get model by ID"""
        return db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    
    @staticmethod
    def set_active_model(db: Session, model_id: int):
        """Set model as active"""
        # Deactivate all models
        db.query(ModelVersion).update({"is_active": False})
        
        # Activate specified model
        model = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
        if model:
            model.is_active = True
            db.commit()


class PredictionCRUD:
    """CRUD operations for predictions"""
    
    @staticmethod
    def create_prediction(db: Session, model_version_id: int, filename: str,
                         predicted_class: str, predicted_class_idx: int, 
                         confidence: float, processing_time_ms: Optional[float] = None,
                         all_probabilities: Optional[Dict] = None,
                         metadata: Optional[Dict] = None) -> Prediction:
        """Create prediction record"""
        db_prediction = Prediction(
            model_version_id=model_version_id,
            filename=filename,
            predicted_class=predicted_class,
            predicted_class_idx=predicted_class_idx,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            all_probabilities=json.dumps(all_probabilities) if all_probabilities else None,
            file_metadata=json.dumps(metadata) if metadata else None
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        return db_prediction
    
    @staticmethod
    def get_predictions(db: Session, skip: int = 0, limit: int = 100,
                       model_version_id: Optional[int] = None) -> List[Prediction]:
        """Get predictions"""
        query = db.query(Prediction)
        if model_version_id:
            query = query.filter(Prediction.model_version_id == model_version_id)
        return query.offset(skip).limit(limit).all()
    
    @staticmethod
    def get_prediction_stats(db: Session) -> Dict[str, Any]:
        """Get prediction statistics"""
        from sqlalchemy import func
        
        total_predictions = db.query(func.count(Prediction.id)).scalar()
        avg_confidence = db.query(func.avg(Prediction.confidence)).scalar() or 0
        
        # Class distribution
        class_dist = db.query(
            Prediction.predicted_class, 
            func.count(Prediction.predicted_class)
        ).group_by(Prediction.predicted_class).all()
        
        # Predictions today
        today = datetime.now().date()
        predictions_today = db.query(func.count(Prediction.id)).filter(
            func.date(Prediction.prediction_timestamp) == today
        ).scalar()
        
        return {
            "total_predictions": total_predictions,
            "average_confidence": float(avg_confidence),
            "class_distribution": dict(class_dist),
            "predictions_today": predictions_today
        }


class RetrainingCRUD:
    """CRUD operations for retraining tasks"""
    
    @staticmethod
    def create_retraining_task(db: Session, task_id: str, dataset_id: int,
                              base_model_id: Optional[int] = None,
                              training_params: Optional[Dict] = None) -> RetrainingTask:
        """Create retraining task"""
        db_task = RetrainingTask(
            task_id=task_id,
            dataset_id=dataset_id,
            base_model_id=base_model_id,
            training_params=json.dumps(training_params) if training_params else None
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    
    @staticmethod
    def update_retraining_task(db: Session, task_id: str, status: str,
                              error_message: Optional[str] = None,
                              new_model_id: Optional[int] = None):
        """Update retraining task"""
        task = db.query(RetrainingTask).filter(RetrainingTask.task_id == task_id).first()
        if task:
            task.status = status
            task.error_message = error_message
            task.new_model_id = new_model_id
            if status in ["completed", "failed"]:
                task.completed_at = datetime.now()
            db.commit()
    
    @staticmethod
    def get_retraining_tasks(db: Session, skip: int = 0, limit: int = 50) -> List[RetrainingTask]:
        """Get retraining tasks"""
        return db.query(RetrainingTask).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_retraining_task(db: Session, task_id: str) -> Optional[RetrainingTask]:
        """Get retraining task by ID"""
        return db.query(RetrainingTask).filter(RetrainingTask.task_id == task_id).first()


class SystemMetricCRUD:
    """CRUD operations for system metrics"""
    
    @staticmethod
    def create_metric(db: Session, metric_name: str, metric_value: float,
                     metadata: Optional[Dict] = None):
        """Create system metric"""
        db_metric = SystemMetric(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_metadata=json.dumps(metadata) if metadata else None
        )
        db.add(db_metric)
        db.commit()
    
    @staticmethod
    def get_metrics(db: Session, metric_name: Optional[str] = None,
                   skip: int = 0, limit: int = 100) -> List[SystemMetric]:
        """Get system metrics"""
        query = db.query(SystemMetric)
        if metric_name:
            query = query.filter(SystemMetric.metric_name == metric_name)
        return query.offset(skip).limit(limit).all()


def get_system_stats(db: Session) -> Dict[str, Any]:
    """Get overall system statistics"""
    total_datasets = db.query(func.count(UploadedDataset.id)).scalar()
    total_models = db.query(func.count(ModelVersion.id)).scalar()
    total_retraining_jobs = db.query(func.count(RetrainingTask.id)).scalar()
    
    prediction_stats = PredictionCRUD.get_prediction_stats(db)
    
    return {
        "total_predictions": prediction_stats["total_predictions"],
        "total_datasets": total_datasets,
        "total_models": total_models,
        "total_retraining_jobs": total_retraining_jobs,
        "average_confidence": prediction_stats["average_confidence"],
        "predictions_today": prediction_stats["predictions_today"]
    }
