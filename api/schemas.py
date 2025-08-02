"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """Status enumeration"""
    uploaded = "uploaded"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    training = "training"
    started = "started"


# Dataset Schemas
class DatasetUploadResponse(BaseModel):
    """Response for dataset upload"""
    id: int
    filename: str
    upload_timestamp: datetime
    total_files: int
    valid_images: int
    classes_info: Dict[str, Any]
    status: str
    
    class Config:
        from_attributes = True


class DatasetInfo(BaseModel):
    """Dataset information schema"""
    id: int
    filename: str
    upload_timestamp: datetime
    file_path: str
    extracted_path: Optional[str] = None
    total_files: int
    valid_images: int
    classes_info: Dict[str, Any]
    status: str
    
    class Config:
        from_attributes = True


# Model Version Schemas
class ModelVersionCreate(BaseModel):
    """Schema for creating model version"""
    model_name: str
    version: str
    model_path: str
    base_model: str
    num_classes: int
    training_params: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    training_dataset_id: Optional[int] = None


class ModelVersionResponse(BaseModel):
    """Response for model version"""
    id: int
    model_name: str
    version: str
    created_timestamp: datetime
    model_path: str
    base_model: str
    num_classes: int
    training_params: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    is_active: bool
    status: str
    
    class Config:
        from_attributes = True


# Prediction Schemas
class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    filename: str
    image_data: Optional[str] = None  # Base64 encoded image data


class PredictionResult(BaseModel):
    """Single prediction result"""
    class_name: str = Field(alias="class")
    confidence: float
    class_index: int


class PredictionResponse(BaseModel):
    """Response for prediction"""
    filename: str
    predictions: List[PredictionResult]
    processing_time_ms: Optional[float] = None
    timestamp: datetime
    
    class Config:
        populate_by_name = True


class PredictionLog(BaseModel):
    """Prediction log schema"""
    id: int
    filename: str
    predicted_class: str
    confidence: float
    prediction_timestamp: datetime
    processing_time_ms: Optional[float] = None
    model_name: str
    model_version: str
    
    class Config:
        from_attributes = True


# Retraining Schemas
class RetrainingRequest(BaseModel):
    """Schema for retraining request"""
    dataset_id: int
    base_model_id: Optional[int] = None
    training_params: Optional[Dict[str, Any]] = None


class RetrainingTaskResponse(BaseModel):
    """Response for retraining task"""
    id: int
    task_id: str
    dataset_id: int
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    training_params: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# System Schemas
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    model_loaded: bool
    database_ready: bool
    active_model: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information schema"""
    model_architecture: str
    input_shape: List[int]
    num_classes: int
    preprocessing: str
    version: Optional[str] = None


class SystemStats(BaseModel):
    """System statistics schema"""
    total_predictions: int
    total_datasets: int
    total_models: int
    total_retraining_jobs: int
    average_confidence: float
    predictions_today: int


class AnalyticsResponse(BaseModel):
    """Analytics response schema"""
    prediction_stats: SystemStats
    class_distribution: Dict[str, int]
    hourly_predictions: Dict[str, int]
    model_performance: Dict[str, Any]


# File Upload Schemas
class FileUploadResponse(BaseModel):
    """File upload response"""
    filename: str
    size: int
    category: Optional[str] = None
    upload_timestamp: datetime


class BatchUploadResponse(BaseModel):
    """Batch upload response"""
    message: str
    files: List[FileUploadResponse]
    total_files: int
    timestamp: datetime


# Error Schemas
class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime


# Metrics Schemas
class PredictionLogEntry(BaseModel):
    """Single prediction log entry"""
    timestamp: str
    filename: str
    predicted_class: str
    confidence: float
    processing_time_ms: float


class ErrorLogEntry(BaseModel):
    """Single error log entry"""
    timestamp: str
    error: str


class SystemMetricsResponse(BaseModel):
    """System metrics response"""
    uptime_start: datetime
    uptime_hours: float
    uptime_formatted: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    total_uploads: int
    total_retraining_jobs: int
    average_processing_time: float
    model_loads: int
    success_rate: float
    failure_rate: float
    recent_predictions: List[PredictionLogEntry]
    recent_errors: List[ErrorLogEntry]
    predictor_loaded: bool
    current_time: str
