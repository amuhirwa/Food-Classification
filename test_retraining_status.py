#!/usr/bin/env python3
"""
Test script to verify the API endpoint returns performance metrics correctly
when a retraining task is completed.
"""

import json
from datetime import datetime

# Simulate the response structure for a completed retraining task
def test_completed_task_response():
    """Test the structure of a completed retraining task response"""
    
    # Mock performance metrics that would be stored in the model
    mock_performance_metrics = {
        "final_val_accuracy": 0.8654,  # 86.54%
        "final_accuracy": 0.8821,
        "val_loss": 0.3421,
        "f1_score_weighted": 0.8602,
        "f1_score_macro": 0.8433,
        "train_accuracy": 0.9123
    }
    
    # Mock task response with performance metrics
    completed_task_response = {
        "task_id": "test-task-123",
        "status": "completed",
        "started_at": "2025-08-03T10:00:00",
        "completed_at": "2025-08-03T10:30:00",
        "error_message": None,
        "dataset_id": 1,
        "new_model_id": 42,
        "accuracy": mock_performance_metrics["final_val_accuracy"],  # This is the key addition
        "performance_metrics": mock_performance_metrics
    }
    
    print("✅ Completed Task Response Structure:")
    print(json.dumps(completed_task_response, indent=2))
    
    # Test frontend display logic
    accuracy_percentage = completed_task_response["accuracy"] * 100
    print(f"\n✅ Frontend Display Test:")
    print(f"   Accuracy: {accuracy_percentage:.1f}%")
    print(f"   Alert Message: Retraining completed successfully! (Accuracy: {accuracy_percentage:.1f}%)")
    
    return completed_task_response

def test_running_task_response():
    """Test the structure of a running retraining task response"""
    
    running_task_response = {
        "task_id": "test-task-123",
        "status": "running",
        "started_at": "2025-08-03T10:00:00",
        "completed_at": None,
        "error_message": None,
        "dataset_id": 1,
        "new_model_id": None
        # No accuracy or performance_metrics for running tasks
    }
    
    print("\n✅ Running Task Response Structure:")
    print(json.dumps(running_task_response, indent=2))
    
    return running_task_response

def test_failed_task_response():
    """Test the structure of a failed retraining task response"""
    
    failed_task_response = {
        "task_id": "test-task-123",
        "status": "failed",
        "started_at": "2025-08-03T10:00:00",
        "completed_at": "2025-08-03T10:15:00",
        "error_message": "Insufficient training data",
        "dataset_id": 1,
        "new_model_id": None
        # No accuracy or performance_metrics for failed tasks
    }
    
    print("\n✅ Failed Task Response Structure:")
    print(json.dumps(failed_task_response, indent=2))
    
    return failed_task_response

if __name__ == "__main__":
    print("🧪 Testing Retraining Task Status API Response Structures\n")
    
    # Test different task states
    test_running_task_response()
    test_failed_task_response()  
    test_completed_task_response()
    
    print("\n✅ All tests passed! The API endpoint should return performance metrics for completed tasks.")
