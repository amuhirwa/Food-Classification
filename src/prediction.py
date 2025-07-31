"""
Prediction module for food classification
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import joblib
import logging
from datetime import datetime
from preprocessing import FoodDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodClassificationPredictor:
    """
    Class for making predictions with trained food classification models
    """
    
    def __init__(self, model_path, metadata_path=None, class_mappings_dir=None):
        self.model_path = model_path
        self.model = None
        self.classes = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.preprocessor = FoodDataPreprocessor()
        
        # Load model and metadata
        self.load_model()
        if metadata_path:
            self.load_metadata(metadata_path)
        if class_mappings_dir:
            self.load_class_mappings(class_mappings_dir)
    
    def load_model(self):
        """
        Load the trained model
        """
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def load_metadata(self, metadata_path):
        """
        Load model metadata
        """
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.num_classes = metadata.get('num_classes')
            self.input_shape = metadata.get('input_shape')
            logger.info(f"Metadata loaded: {metadata}")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
    
    def load_class_mappings(self, mappings_dir):
        """
        Load class mappings
        """
        try:
            with open(os.path.join(mappings_dir, 'class_to_idx.json'), 'r') as f:
                self.class_to_idx = json.load(f)
            
            with open(os.path.join(mappings_dir, 'idx_to_class.json'), 'r') as f:
                self.idx_to_class = {int(k): v for k, v in json.load(f).items()}
            
            with open(os.path.join(mappings_dir, 'classes.json'), 'r') as f:
                self.classes = json.load(f)
            
            logger.info(f"Class mappings loaded: {len(self.classes)} classes")
            
        except Exception as e:
            logger.error(f"Error loading class mappings: {str(e)}")
    
    def predict_single_image(self, image_path, return_probabilities=False, top_k=3):
        """
        Predict class for a single image
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Preprocess image
            img_array = self.preprocessor.preprocess_single_image(image_path)
            if img_array is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            result = {
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add class name if mappings available
            if self.idx_to_class:
                result['predicted_class'] = self.idx_to_class.get(predicted_class_idx, 'Unknown')
            
            # Add top-k predictions if requested
            if return_probabilities:
                top_indices = np.argsort(predictions[0])[-top_k:][::-1]
                top_predictions = []
                
                for idx in top_indices:
                    pred_info = {
                        'class_idx': int(idx),
                        'confidence': float(predictions[0][idx])
                    }
                    if self.idx_to_class:
                        pred_info['class_name'] = self.idx_to_class.get(idx, 'Unknown')
                    top_predictions.append(pred_info)
                
                result['top_predictions'] = top_predictions
            
            logger.info(f"Prediction made for {image_path}: {result['predicted_class'] if 'predicted_class' in result else predicted_class_idx}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict classes for multiple images
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            results = []
            
            # Process images in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                valid_paths = []
                
                # Preprocess batch
                for path in batch_paths:
                    img_array = self.preprocessor.preprocess_single_image(path)
                    if img_array is not None:
                        batch_images.append(img_array[0])  # Remove batch dimension
                        valid_paths.append(path)
                
                if batch_images:
                    batch_array = np.array(batch_images)
                    
                    # Make predictions
                    predictions = self.model.predict(batch_array, verbose=0)
                    
                    # Process results
                    for j, path in enumerate(valid_paths):
                        predicted_class_idx = np.argmax(predictions[j])
                        confidence = np.max(predictions[j])
                        
                        result = {
                            'image_path': path,
                            'predicted_class_idx': int(predicted_class_idx),
                            'confidence': float(confidence),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        if self.idx_to_class:
                            result['predicted_class'] = self.idx_to_class.get(predicted_class_idx, 'Unknown')
                        
                        results.append(result)
            
            logger.info(f"Batch prediction completed for {len(results)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return None
    
    def predict_from_bytes(self, image_bytes, return_probabilities=False):
        """
        Predict class for image from bytes (for API usage)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        try:
            # Preprocess image from bytes
            img_array = self.preprocessor.preprocess_image_from_bytes(image_bytes)
            if img_array is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            result = {
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add class name if mappings available
            if self.idx_to_class:
                result['predicted_class'] = self.idx_to_class.get(predicted_class_idx, 'Unknown')
            
            # Add all class probabilities if requested
            if return_probabilities:
                probabilities = {}
                for idx, prob in enumerate(predictions[0]):
                    class_name = self.idx_to_class.get(idx, f'Class_{idx}') if self.idx_to_class else f'Class_{idx}'
                    probabilities[class_name] = float(prob)
                result['all_probabilities'] = probabilities
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction from bytes: {str(e)}")
            return None
    
    def validate_model_performance(self, test_data_dir):
        """
        Validate model performance on test data
        """
        try:
            # Create test data generator
            test_gen, _ = self.preprocessor.create_data_generators(
                test_data_dir, validation_split=0.0, augment=False
            )
            
            # Make predictions
            test_gen.reset()
            predictions = self.model.predict(test_gen, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Get true labels
            true_classes = test_gen.classes
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            accuracy = accuracy_score(true_classes, predicted_classes)
            report = classification_report(true_classes, predicted_classes, 
                                         target_names=list(test_gen.class_indices.keys()),
                                         output_dict=True)
            
            validation_results = {
                'accuracy': accuracy,
                'classification_report': report,
                'total_samples': len(true_classes),
                'validation_date': datetime.now().isoformat()
            }
            
            logger.info(f"Model validation completed. Accuracy: {accuracy:.4f}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during model validation: {str(e)}")
            return None
    
    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': self.model.count_params(),
            'num_classes': len(self.classes) if self.classes else 'Unknown',
            'classes': self.classes
        }
        
        return info
    
    def explain_prediction(self, image_path, method='gradcam'):
        """
        Generate explanation for prediction (placeholder for interpretability)
        """
        # This would implement model interpretability techniques
        # like GradCAM, LIME, or SHAP
        logger.info(f"Explanation method '{method}' not yet implemented")
        return {"explanation": "Feature not implemented yet"}

class ModelEnsemble:
    """
    Class for ensemble predictions using multiple models
    """
    
    def __init__(self, model_paths, class_mappings_dir):
        self.predictors = []
        
        for model_path in model_paths:
            predictor = FoodClassificationPredictor(
                model_path=model_path,
                class_mappings_dir=class_mappings_dir
            )
            if predictor.model is not None:
                self.predictors.append(predictor)
        
        logger.info(f"Ensemble created with {len(self.predictors)} models")
    
    def predict_ensemble(self, image_path, method='average'):
        """
        Make ensemble prediction
        """
        if not self.predictors:
            return None
        
        try:
            predictions = []
            
            # Get predictions from all models
            for predictor in self.predictors:
                result = predictor.predict_single_image(image_path, return_probabilities=True)
                if result and 'top_predictions' in result:
                    # Extract probability vector
                    prob_vector = np.zeros(len(predictor.classes))
                    for pred in result['top_predictions']:
                        prob_vector[pred['class_idx']] = pred['confidence']
                    predictions.append(prob_vector)
            
            if not predictions:
                return None
            
            # Combine predictions
            if method == 'average':
                ensemble_probs = np.mean(predictions, axis=0)
            elif method == 'max':
                ensemble_probs = np.max(predictions, axis=0)
            else:
                ensemble_probs = np.mean(predictions, axis=0)
            
            # Get final prediction
            predicted_class_idx = np.argmax(ensemble_probs)
            confidence = np.max(ensemble_probs)
            
            result = {
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': float(confidence),
                'ensemble_method': method,
                'num_models': len(self.predictors),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add class name if available
            if self.predictors[0].idx_to_class:
                result['predicted_class'] = self.predictors[0].idx_to_class.get(predicted_class_idx, 'Unknown')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    model_path = "../models/food_classifier.h5"
    class_mappings_dir = "../models"
    
    if os.path.exists(model_path):
        predictor = FoodClassificationPredictor(
            model_path=model_path,
            class_mappings_dir=class_mappings_dir
        )
        
        print("Predictor initialized successfully!")
        print(f"Model info: {predictor.get_model_info()}")
    else:
        print(f"Model not found at {model_path}")
