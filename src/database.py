"""
Database module for MLOps pipeline
Handles data storage, model versioning, and prediction logging
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MLOpsDatabase:
    """
    SQLite database for MLOps pipeline
    """
    
    def __init__(self, db_path: str = "mlops.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Uploaded datasets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        upload_timestamp DATETIME NOT NULL,
                        file_path TEXT NOT NULL,
                        extracted_path TEXT,
                        total_files INTEGER,
                        valid_images INTEGER,
                        classes_info TEXT,
                        status TEXT DEFAULT 'uploaded',
                        metadata TEXT
                    )
                """)
                
                # Model versions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        created_timestamp DATETIME NOT NULL,
                        model_path TEXT NOT NULL,
                        base_model TEXT,
                        num_classes INTEGER,
                        training_params TEXT,
                        performance_metrics TEXT,
                        is_active BOOLEAN DEFAULT FALSE,
                        training_dataset_id INTEGER,
                        status TEXT DEFAULT 'training',
                        FOREIGN KEY (training_dataset_id) REFERENCES uploaded_datasets (id)
                    )
                """)
                
                # Predictions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version_id INTEGER,
                        filename TEXT,
                        predicted_class TEXT,
                        predicted_class_idx INTEGER,
                        confidence REAL,
                        prediction_timestamp DATETIME NOT NULL,
                        processing_time_ms REAL,
                        image_size TEXT,
                        all_probabilities TEXT,
                        metadata TEXT,
                        FOREIGN KEY (model_version_id) REFERENCES model_versions (id)
                    )
                """)
                
                # Retraining tasks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS retraining_tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT UNIQUE NOT NULL,
                        dataset_id INTEGER,
                        base_model_id INTEGER,
                        status TEXT DEFAULT 'started',
                        started_at DATETIME NOT NULL,
                        completed_at DATETIME,
                        error_message TEXT,
                        new_model_id INTEGER,
                        training_params TEXT,
                        FOREIGN KEY (dataset_id) REFERENCES uploaded_datasets (id),
                        FOREIGN KEY (base_model_id) REFERENCES model_versions (id),
                        FOREIGN KEY (new_model_id) REFERENCES model_versions (id)
                    )
                """)
                
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL,
                        metadata TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_uploaded_dataset(self, filename: str, file_path: str, extracted_path: str, 
                            data_info: Dict[str, Any]) -> int:
        """Save uploaded dataset information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO uploaded_datasets 
                    (filename, upload_timestamp, file_path, extracted_path, total_files, 
                     valid_images, classes_info, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filename,
                    datetime.now(),
                    file_path,
                    extracted_path,
                    data_info.get('total_files', 0),
                    data_info.get('valid_images', 0),
                    json.dumps(data_info.get('classes', {})),
                    json.dumps(data_info)
                ))
                
                dataset_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Dataset saved with ID: {dataset_id}")
                return dataset_id
                
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")
            raise
    
    def save_model_version(self, model_name: str, version: str, model_path: str,
                          base_model: str, num_classes: int, training_params: Dict,
                          performance_metrics: Dict = None, training_dataset_id: int = None) -> int:
        """Save new model version"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Deactivate previous active model
                cursor.execute("UPDATE model_versions SET is_active = FALSE")
                
                cursor.execute("""
                    INSERT INTO model_versions 
                    (model_name, version, created_timestamp, model_path, base_model, 
                     num_classes, training_params, performance_metrics, is_active, 
                     training_dataset_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    version,
                    datetime.now(),
                    model_path,
                    base_model,
                    num_classes,
                    json.dumps(training_params),
                    json.dumps(performance_metrics) if performance_metrics else None,
                    True,  # Set as active
                    training_dataset_id,
                    'completed'
                ))
                
                model_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Model version saved with ID: {model_id}")
                return model_id
                
        except Exception as e:
            logger.error(f"Error saving model version: {e}")
            raise
    
    def log_prediction(self, model_version_id: int, filename: str, predicted_class: str,
                      predicted_class_idx: int, confidence: float, processing_time_ms: float = None,
                      all_probabilities: Dict = None, metadata: Dict = None) -> int:
        """Log prediction result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO predictions 
                    (model_version_id, filename, predicted_class, predicted_class_idx, 
                     confidence, prediction_timestamp, processing_time_ms, 
                     all_probabilities, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_version_id,
                    filename,
                    predicted_class,
                    predicted_class_idx,
                    confidence,
                    datetime.now(),
                    processing_time_ms,
                    json.dumps(all_probabilities) if all_probabilities else None,
                    json.dumps(metadata) if metadata else None
                ))
                
                prediction_id = cursor.lastrowid
                conn.commit()
                
                return prediction_id
                
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            raise
    
    def create_retraining_task(self, task_id: str, dataset_id: int, 
                             base_model_id: int = None, training_params: Dict = None) -> int:
        """Create retraining task record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO retraining_tasks 
                    (task_id, dataset_id, base_model_id, started_at, training_params)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    task_id,
                    dataset_id,
                    base_model_id,
                    datetime.now(),
                    json.dumps(training_params) if training_params else None
                ))
                
                task_db_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Retraining task created with ID: {task_db_id}")
                return task_db_id
                
        except Exception as e:
            logger.error(f"Error creating retraining task: {e}")
            raise
    
    def update_retraining_task(self, task_id: str, status: str, 
                             error_message: str = None, new_model_id: int = None):
        """Update retraining task status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                completed_at = datetime.now() if status in ['completed', 'failed'] else None
                
                cursor.execute("""
                    UPDATE retraining_tasks 
                    SET status = ?, completed_at = ?, error_message = ?, new_model_id = ?
                    WHERE task_id = ?
                """, (status, completed_at, error_message, new_model_id, task_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating retraining task: {e}")
            raise
    
    def get_active_model(self) -> Optional[Dict]:
        """Get currently active model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM model_versions 
                    WHERE is_active = TRUE 
                    ORDER BY created_timestamp DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            return None
    
    def get_datasets(self, limit: int = 50) -> List[Dict]:
        """Get uploaded datasets"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM uploaded_datasets 
                    ORDER BY upload_timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting datasets: {e}")
            return []
    
    def get_model_versions(self, limit: int = 50) -> List[Dict]:
        """Get model versions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT mv.*, ud.filename as dataset_filename 
                    FROM model_versions mv
                    LEFT JOIN uploaded_datasets ud ON mv.training_dataset_id = ud.id
                    ORDER BY mv.created_timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting model versions: {e}")
            return []
    
    def get_predictions(self, limit: int = 100, model_version_id: int = None) -> List[Dict]:
        """Get prediction logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if model_version_id:
                    cursor.execute("""
                        SELECT p.*, mv.model_name, mv.version
                        FROM predictions p
                        JOIN model_versions mv ON p.model_version_id = mv.id
                        WHERE p.model_version_id = ?
                        ORDER BY p.prediction_timestamp DESC 
                        LIMIT ?
                    """, (model_version_id, limit))
                else:
                    cursor.execute("""
                        SELECT p.*, mv.model_name, mv.version
                        FROM predictions p
                        JOIN model_versions mv ON p.model_version_id = mv.id
                        ORDER BY p.prediction_timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []
    
    def get_retraining_tasks(self, limit: int = 50) -> List[Dict]:
        """Get retraining tasks"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT rt.*, ud.filename as dataset_filename,
                           bmv.model_name as base_model_name,
                           nmv.model_name as new_model_name
                    FROM retraining_tasks rt
                    LEFT JOIN uploaded_datasets ud ON rt.dataset_id = ud.id
                    LEFT JOIN model_versions bmv ON rt.base_model_id = bmv.id
                    LEFT JOIN model_versions nmv ON rt.new_model_id = nmv.id
                    ORDER BY rt.started_at DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Error getting retraining tasks: {e}")
            return []
    
    def get_prediction_analytics(self) -> Dict:
        """Get prediction analytics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total_predictions = cursor.fetchone()[0]
                
                # Predictions by class
                cursor.execute("""
                    SELECT predicted_class, COUNT(*) as count
                    FROM predictions 
                    GROUP BY predicted_class
                    ORDER BY count DESC
                """)
                class_distribution = dict(cursor.fetchall())
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Predictions over time (last 24 hours)
                cursor.execute("""
                    SELECT strftime('%H', prediction_timestamp) as hour, COUNT(*) as count
                    FROM predictions 
                    WHERE prediction_timestamp >= datetime('now', '-24 hours')
                    GROUP BY hour
                    ORDER BY hour
                """)
                hourly_predictions = dict(cursor.fetchall())
                
                return {
                    "total_predictions": total_predictions,
                    "class_distribution": class_distribution,
                    "average_confidence": avg_confidence,
                    "hourly_predictions": hourly_predictions
                }
                
        except Exception as e:
            logger.error(f"Error getting prediction analytics: {e}")
            return {}
    
    def save_system_metric(self, metric_name: str, metric_value: float, metadata: Dict = None):
        """Save system metric"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    datetime.now(),
                    metric_name,
                    metric_value,
                    json.dumps(metadata) if metadata else None
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving system metric: {e}")

# Global database instance
db = MLOpsDatabase()

if __name__ == "__main__":
    # Test database functionality
    print("Testing MLOps Database...")
    
    test_db = MLOpsDatabase("test_mlops.db")
    
    # Test dataset upload
    dataset_id = test_db.save_uploaded_dataset(
        "test_dataset.zip",
        "/path/to/dataset.zip",
        "/path/to/extracted",
        {"total_files": 100, "valid_images": 95, "classes": {"bread": 50, "meat": 45}}
    )
    print(f"Dataset saved with ID: {dataset_id}")
    
    # Test model version
    model_id = test_db.save_model_version(
        "food_classifier",
        "v1.0",
        "/path/to/model.h5",
        "MobileNetV2",
        11,
        {"epochs": 50, "batch_size": 32}
    )
    print(f"Model saved with ID: {model_id}")
    
    # Test prediction logging
    pred_id = test_db.log_prediction(
        model_id,
        "test_image.jpg",
        "bread",
        0,
        0.95
    )
    print(f"Prediction logged with ID: {pred_id}")
    
    print("Database test completed!")
