"""
Training script for food classification model
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import FoodDataPreprocessor
from model import FoodClassificationModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train food classification model')
    parser.add_argument('--data-dir', default='../data', help='Directory containing training data')
    parser.add_argument('--model-dir', default='../models', help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--base-model', default='MobileNetV2', choices=['MobileNetV2', 'EfficientNetB0', 'ResNet50'])
    parser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning of base model')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    logger.info(f"Starting training with parameters: {args}")
    
    # Initialize preprocessor
    preprocessor = FoodDataPreprocessor(batch_size=args.batch_size)
    
    # Load and prepare data
    logger.info("Loading and preprocessing data...")
    classes = preprocessor.load_classes_from_directory(args.data_dir)
    train_gen, val_gen = preprocessor.create_data_generators(
        args.data_dir, 
        validation_split=args.validation_split
    )
    
    # Analyze dataset
    stats = preprocessor.analyze_dataset(args.data_dir)
    logger.info(f"Dataset statistics: {stats}")
    
    # Save class mappings
    preprocessor.save_class_mappings(args.model_dir)
    
    # Initialize model
    logger.info(f"Creating model with {len(classes)} classes...")
    food_model = FoodClassificationModel(
        num_classes=len(classes),
        model_dir=args.model_dir
    )
    
    # Create model
    model = food_model.create_model(
        base_model_name=args.base_model,
        fine_tune=args.fine_tune
    )
    
    if model is None:
        logger.error("Failed to create model")
        return
    
    logger.info(f"Model created successfully. Total parameters: {model.count_params():,}")
    
    # Train model
    logger.info("Starting training...")
    history = food_model.train_model(
        train_gen,
        val_gen,
        epochs=args.epochs
    )
    
    if history is None:
        logger.error("Training failed")
        return
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluation_results = food_model.evaluate_model(val_gen)
    
    if evaluation_results:
        logger.info(f"Evaluation results: {evaluation_results}")
    
    # Save model with performance metrics
    logger.info("Saving model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"food_classifier_{args.base_model}_{timestamp}"
    
    success = food_model.save_model(model_name, include_metadata=True, validation_generator=val_gen)
    
    if success:
        logger.info(f"Model saved successfully as {model_name}")
        
        # Also save as final model for deployment with metrics
        final_model_name = "food_classifier_final"
        food_model.save_model(final_model_name, include_metadata=True, validation_generator=val_gen)
        logger.info(f"Model also saved as {final_model_name} for deployment")
    else:
        logger.error("Failed to save model")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
