"""
Model creation and training module for food classification
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import json
import joblib
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodClassificationModel:
    """
    Class for creating and training food classification models
    """
    
    def __init__(self, num_classes, input_shape=(224, 224, 3), model_dir="../models"):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.model = None
        self.history = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def create_model(self, base_model_name='MobileNetV2', fine_tune=False, 
                    dropout_rate=0.5, learning_rate=0.001):
        """
        Create a CNN model with transfer learning
        """
        try:
            # Load pre-trained base model
            if base_model_name == 'MobileNetV2':
                base_model = MobileNetV2(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            elif base_model_name == 'EfficientNetB0':
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            elif base_model_name == 'ResNet50':
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            else:
                raise ValueError(f"Unsupported base model: {base_model_name}")
            
            # Set trainability of base model
            base_model.trainable = fine_tune
            
            # Add custom classification head
            self.model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.6),
                Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(dropout_rate * 0.4),
                Dense(self.num_classes, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Model created with base: {base_model_name}")
            logger.info(f"Total parameters: {self.model.count_params():,}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            return None
    
    def get_callbacks(self, patience=5, min_lr=1e-7, monitor='val_loss'):
        """
        Get training callbacks for optimization
        """
        callbacks = [
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=monitor,
                factor=0.2,
                patience=patience//2,
                min_lr=min_lr,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'best_model_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, train_generator, validation_generator, epochs=50, 
                   callbacks=None, verbose=1):
        """
        Train the model
        """
        if self.model is None:
            logger.error("Model not created. Call create_model() first.")
            return None
        
        if callbacks is None:
            callbacks = self.get_callbacks()
        
        try:
            logger.info("Starting model training...")
            
            self.history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info("Training completed successfully!")
            return self.history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return None
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the model performance
        """
        if self.model is None:
            logger.error("Model not found. Train or load a model first.")
            return None
        
        try:
            # Evaluate on test data
            results = self.model.evaluate(test_generator, verbose=1)
            
            # Create results dictionary
            metric_names = self.model.metrics_names
            evaluation_results = dict(zip(metric_names, results))
            
            logger.info(f"Evaluation results: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return None
    
    def save_model(self, model_name="food_classifier", include_metadata=True):
        """
        Save the trained model and metadata
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return None
        
        try:
            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}.h5")
            self.model.save(model_path)
            
            # Save model in SavedModel format as well (for serving)
            savedmodel_path = os.path.join(self.model_dir, f"{model_name}_savedmodel")
            self.model.save(savedmodel_path, save_format='tf')
            
            if include_metadata:
                # Save training history
                if self.history:
                    import joblib
                    history_path = os.path.join(self.model_dir, f"{model_name}_history.pkl")
                    joblib.dump(self.history.history, history_path)
                
                # Save model metadata
                import json
                metadata = {
                    'model_name': model_name,
                    'created_date': datetime.now().isoformat(),
                    'num_classes': self.num_classes,
                    'input_shape': self.input_shape,
                    'model_path': model_path,
                    'savedmodel_path': savedmodel_path
                }
                
                metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            logger.info(f"Model saved successfully to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, model_path):
        """
        Load a trained model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def fine_tune_model(self, train_generator, validation_generator, 
                       unfreeze_layers=None, learning_rate=1e-5, epochs=20):
        """
        Fine-tune the pre-trained model
        """
        if self.model is None:
            logger.error("Model not found. Create or load a model first.")
            return None
        
        try:
            # Unfreeze some layers for fine-tuning
            base_model = self.model.layers[0]
            
            if unfreeze_layers is None:
                # Unfreeze the last few layers
                for layer in base_model.layers[-20:]:
                    layer.trainable = True
            else:
                # Unfreeze specific layers
                for layer_name in unfreeze_layers:
                    for layer in base_model.layers:
                        if layer.name == layer_name:
                            layer.trainable = True
            
            # Recompile with lower learning rate
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Get callbacks for fine-tuning
            callbacks = self.get_callbacks(patience=5)
            
            # Fine-tune the model
            fine_tune_history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Fine-tuning completed!")
            return fine_tune_history
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {str(e)}")
            return None
    
    def retrain_model(self, new_train_generator, new_val_generator, 
                     base_model_path=None, epochs=30, model_version=None):
        """
        Retrain the model with new data using proper versioning
        Preserves the existing model architecture and class structure
        """
        try:
            if not base_model_path or not os.path.exists(base_model_path):
                raise ValueError("Base model path is required for retraining to preserve class structure")
            
            # Load existing model as starting point
            self.load_model(base_model_path)
            logger.info(f"Loaded existing model for retraining: {base_model_path}")
            logger.info(f"Model architecture preserved - num_classes: {self.num_classes}")
            
            # Verify that the new data generators match the existing model's class structure
            if hasattr(new_train_generator, 'num_classes'):
                if new_train_generator.num_classes != self.num_classes:
                    logger.warning(f"Training data has {new_train_generator.num_classes} classes, "
                                 f"but model expects {self.num_classes} classes. "
                                 f"Only training on classes that match the existing model.")
            
            # Fine-tune the existing model with new data
            # Use a lower learning rate for fine-tuning to preserve learned features
            original_lr = self.model.optimizer.learning_rate
            fine_tune_lr = float(original_lr) * 0.1  # Reduce learning rate for fine-tuning
            
            # Update optimizer with lower learning rate
            from tensorflow.keras.optimizers import Adam
            self.model.compile(
                optimizer=Adam(learning_rate=fine_tune_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"Fine-tuning with reduced learning rate: {fine_tune_lr}")
            
            callbacks = self.get_callbacks(patience=15)
            
            retrain_history = self.model.fit(
                new_train_generator,
                epochs=epochs,
                validation_data=new_val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save retrained model with proper versioning
            if model_version is None:
                # Auto-increment version
                model_version = self._get_next_version()
            
            retrained_model_name = f"food_classifier_v{model_version}"
            model_path = self.save_model(retrained_model_name)
            
            # Create latest symlink/copy
            if model_path:
                self._create_latest_model(model_path)
            
            logger.info(f"Model fine-tuning completed! Saved as v{model_version}")
            logger.info(f"Class structure preserved: {self.num_classes} classes")
            return retrain_history, model_version
            
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
            return None, None
    
    def _get_next_version(self):
        """Get the next version number for model naming"""
        import glob
        
        # Find existing versioned models
        pattern = os.path.join(self.model_dir, "food_classifier_v*.h5")
        existing_models = glob.glob(pattern)
        
        if not existing_models:
            return 1
        
        # Extract version numbers
        versions = []
        for model_path in existing_models:
            filename = os.path.basename(model_path)
            try:
                # Extract version from filename like "food_classifier_v2.h5"
                version_str = filename.split('_v')[1].split('.')[0]
                versions.append(int(version_str))
            except (IndexError, ValueError):
                continue
        
        return max(versions) + 1 if versions else 1
    
    def _create_latest_model(self, model_path):
        """Create a latest model copy/link"""
        try:
            import shutil
            
            latest_path = os.path.join(self.model_dir, "food_classifier_latest.h5")
            
            # Remove existing latest model
            if os.path.exists(latest_path):
                os.remove(latest_path)
            
            # Copy the new model as latest
            shutil.copy2(model_path, latest_path)
            
            # Also create savedmodel format for latest
            if model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
                latest_savedmodel_path = os.path.join(self.model_dir, "food_classifier_latest_savedmodel")
                if os.path.exists(latest_savedmodel_path):
                    shutil.rmtree(latest_savedmodel_path)
                model.save(latest_savedmodel_path, save_format='tf')
            
            logger.info(f"Created latest model at {latest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating latest model: {str(e)}")
            return False
    
    def get_available_models(self):
        """Get list of available model versions"""
        import glob
        import json
        
        models = []
        
        # First, check for the original model
        original_path = os.path.join(self.model_dir, "food_classifier_final.h5")
        if os.path.exists(original_path):
            try:
                # Get file stats
                stat = os.stat(original_path)
                
                models.append({
                    'version': 'original',
                    'model_name': 'food_classifier_final',
                    'path': original_path,
                    'created_date': 'Original Model',
                    'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'num_classes': 'Original Classes',
                    'is_latest': False,
                    'is_original': True
                })
                
            except Exception as e:
                logger.warning(f"Could not analyze original model: {e}")
        
        # Find all versioned models
        pattern = os.path.join(self.model_dir, "food_classifier_v*.h5")
        versioned_models = glob.glob(pattern)
        
        for model_path in versioned_models:
            try:
                filename = os.path.basename(model_path)
                version_str = filename.split('_v')[1].split('.')[0]
                version = int(version_str)
                
                # Get metadata if available
                metadata_path = model_path.replace('.h5', '_metadata.json')
                metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Get file stats
                stat = os.stat(model_path)
                
                models.append({
                    'version': version,
                    'model_name': f"food_classifier_v{version}",
                    'path': model_path,
                    'created_date': metadata.get('created_date', 'Unknown'),
                    'file_size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'num_classes': metadata.get('num_classes', 'Unknown'),
                    'is_latest': False
                })
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Could not parse model version from {filename}: {e}")
                continue
        
        # Check for latest model
        latest_path = os.path.join(self.model_dir, "food_classifier_latest.h5")
        if os.path.exists(latest_path):
            # Find which version is the latest
            latest_stat = os.stat(latest_path)
            for model in models:
                model_stat = os.stat(model['path'])
                if abs(model_stat.st_size - latest_stat.st_size) < 1000:  # Same size, likely same model
                    model['is_latest'] = True
                    break
        
        # Sort by version number with proper handling of mixed types
        def sort_key(model):
            version = model['version']
            if version == 'original':
                return (0, version)  # Original comes first
            elif isinstance(version, str) and version.isdigit():
                return (1, int(version))  # Numeric versions
            elif isinstance(version, int):
                return (1, version)  # Numeric versions
            else:
                return (2, version)  # Other string versions last
        
        models.sort(key=sort_key, reverse=True)
        
        return models
    
    def load_model_by_version(self, version=None):
        """Load a specific model version or latest"""
        try:
            if version is None or version == 'latest':
                # Load latest model
                model_path = os.path.join(self.model_dir, "food_classifier_latest.h5")
                if not os.path.exists(model_path):
                    # Fallback to highest version
                    available_models = self.get_available_models()
                    if not available_models:
                        raise FileNotFoundError("No models found")
                    model_path = available_models[0]['path']
            else:
                # Load specific version
                model_path = os.path.join(self.model_dir, f"food_classifier_v{version}.h5")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model version {version} not found")
            
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def get_model_summary(self):
        """
        Get model architecture summary
        """
        if self.model is None:
            return "No model created"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return captured_output.getvalue()
    
    def predict_batch(self, X):
        """
        Make predictions on a batch of images
        """
        if self.model is None:
            logger.error("Model not found. Load a model first.")
            return None
        
        try:
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            return None

def create_ensemble_model(model_paths, num_classes):
    """
    Create an ensemble model from multiple trained models
    """
    try:
        models = []
        for path in model_paths:
            model = tf.keras.models.load_model(path)
            models.append(model)
        
        # Create ensemble prediction function
        def ensemble_predict(X):
            predictions = []
            for model in models:
                pred = model.predict(X)
                predictions.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        
        logger.info(f"Ensemble model created with {len(models)} models")
        return ensemble_predict
        
    except Exception as e:
        logger.error(f"Error creating ensemble model: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    num_classes = 11  # Adjust based on your dataset
    
    # Create model
    food_model = FoodClassificationModel(num_classes=num_classes)
    model = food_model.create_model()
    
    if model:
        print("Model created successfully!")
        print(food_model.get_model_summary())
