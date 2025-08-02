"""
SQLAlchemy database models for MLOps pipeline
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class UploadedDataset(Base):
    """Model for uploaded datasets"""
    __tablename__ = "uploaded_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, default=func.now())
    file_path = Column(String, nullable=False)
    extracted_path = Column(String)
    total_files = Column(Integer, default=0)
    valid_images = Column(Integer, default=0)
    classes_info = Column(Text)  # JSON string
    status = Column(String, default="uploaded")
    file_metadata = Column(Text)  # JSON string
    
    # Relationships
    model_versions = relationship("ModelVersion", back_populates="training_dataset")
    retraining_tasks = relationship("RetrainingTask", back_populates="dataset")


class ModelVersion(Base):
    """Model for model versions"""
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    created_timestamp = Column(DateTime, default=func.now())
    model_path = Column(String, nullable=False)
    base_model = Column(String)
    num_classes = Column(Integer)
    training_params = Column(Text)  # JSON string
    performance_metrics = Column(Text)  # JSON string
    is_active = Column(Boolean, default=False)
    training_dataset_id = Column(Integer, ForeignKey("uploaded_datasets.id"))
    status = Column(String, default="training")
    
    # Relationships
    training_dataset = relationship("UploadedDataset", back_populates="model_versions")
    predictions = relationship("Prediction", back_populates="model_version")
    base_retraining_tasks = relationship("RetrainingTask", back_populates="base_model", foreign_keys="RetrainingTask.base_model_id")
    new_retraining_tasks = relationship("RetrainingTask", back_populates="new_model", foreign_keys="RetrainingTask.new_model_id")


class Prediction(Base):
    """Model for predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    filename = Column(String)
    predicted_class = Column(String)
    predicted_class_idx = Column(Integer)
    confidence = Column(Float)
    prediction_timestamp = Column(DateTime, default=func.now())
    processing_time_ms = Column(Float)
    image_size = Column(String)
    all_probabilities = Column(Text)  # JSON string
    file_metadata = Column(Text)  # JSON string
    
    # Relationships
    model_version = relationship("ModelVersion", back_populates="predictions")


class RetrainingTask(Base):
    """Model for retraining tasks"""
    __tablename__ = "retraining_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, nullable=False)
    dataset_id = Column(Integer, ForeignKey("uploaded_datasets.id"))
    base_model_id = Column(Integer, ForeignKey("model_versions.id"))
    status = Column(String, default="started")
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    error_message = Column(Text)
    new_model_id = Column(Integer, ForeignKey("model_versions.id"))
    training_params = Column(Text)  # JSON string
    
    # Relationships
    dataset = relationship("UploadedDataset", back_populates="retraining_tasks")
    base_model = relationship("ModelVersion", back_populates="base_retraining_tasks", foreign_keys=[base_model_id])
    new_model = relationship("ModelVersion", back_populates="new_retraining_tasks", foreign_keys=[new_model_id])


class SystemMetric(Base):
    """Model for system metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=func.now())
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float)
    metric_metadata = Column(Text)  # JSON string
