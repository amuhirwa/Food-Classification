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
    
    def get_callbacks(self, patience=10, min_lr=1e-7, monitor='val_loss'):
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
            return False
        
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
                    history_path = os.path.join(self.model_dir, f"{model_name}_history.pkl")
                    joblib.dump(self.history.history, history_path)
                
                # Save model metadata
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
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
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
                     base_model_path=None, epochs=30):
        """
        Retrain the model with new data
        """
        try:
            if base_model_path and os.path.exists(base_model_path):
                # Load existing model as starting point
                self.load_model(base_model_path)
                logger.info("Loaded existing model for retraining")
            elif self.model is None:
                # Create new model if none exists
                logger.info("Creating new model for retraining")
                self.create_model()
            
            # Train with new data
            callbacks = self.get_callbacks(patience=15)
            
            retrain_history = self.model.fit(
                new_train_generator,
                epochs=epochs,
                validation_data=new_val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save retrained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            retrained_model_name = f"food_classifier_retrained_{timestamp}"
            self.save_model(retrained_model_name)
            
            logger.info("Model retraining completed!")
            return retrain_history
            
        except Exception as e:
            logger.error(f"Error during retraining: {str(e)}")
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
