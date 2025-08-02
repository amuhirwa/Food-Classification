"""
MLOps API Package
"""

from .database import get_db, DatasetCRUD, ModelCRUD, PredictionCRUD, RetrainingCRUD
from .models import UploadedDataset, ModelVersion, Prediction, RetrainingTask, SystemMetric
from .schemas import (
    DatasetUploadResponse, ModelVersionResponse, PredictionResponse,
    RetrainingTaskResponse, HealthResponse, ModelInfo, SystemStats
)

__version__ = "2.0.0"
