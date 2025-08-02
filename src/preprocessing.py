"""
Data preprocessing module for food classification
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodDataPreprocessor:
    """
    Class for preprocessing food classification data
    """
    
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = None
        self.class_to_idx = None
        self.idx_to_class = None
        
    def load_classes_from_directory(self, data_dir):
        """
        Load class names from directory structure
        """
        self.classes = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d))]
        self.classes.sort()
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        logger.info(f"Loaded {len(self.classes)} classes: {self.classes}")
        return self.classes
    
    def create_data_generators(self, data_dir, validation_split=0.2, augment=True, preserve_class_mappings=False):
        """
        Create training and validation data generators
        
        Args:
            data_dir: Directory containing training data
            validation_split: Fraction of data to use for validation
            augment: Whether to apply data augmentation
            preserve_class_mappings: If True, preserve existing class mappings instead of updating from data
        """
        if augment:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=validation_split
            )
        else:
            # No augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
        
        # Only rescaling for validation data
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # If preserving class mappings, create a custom classes list based on existing mappings
        classes_list = None
        if preserve_class_mappings and hasattr(self, 'classes') and self.classes:
            classes_list = self.classes
            logger.info(f"Preserving existing class mappings: {classes_list}")
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42,
            classes=classes_list  # Use existing class order if preserving mappings
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42,
            classes=classes_list  # Use existing class order if preserving mappings
        )
        
        # Update class mappings from generator only if not preserving existing ones
        if not preserve_class_mappings:
            self.class_to_idx = train_generator.class_indices
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            self.classes = list(self.class_to_idx.keys())
            logger.info(f"Updated class mappings from data: {self.classes}")
        else:
            logger.info(f"Preserved existing class mappings: {self.classes}")
            # Verify that the generator's class indices match our preserved mappings
            if train_generator.class_indices != self.class_to_idx:
                logger.warning(f"Generator class indices {train_generator.class_indices} "
                             f"differ from preserved mappings {self.class_to_idx}")
        
        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {validation_generator.samples}")
        
        return train_generator, validation_generator
    
    def preprocess_single_image(self, image_path):
        """
        Preprocess a single image for prediction
        """
        try:
            # Load image
            img = load_img(image_path, target_size=self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def preprocess_image_from_bytes(self, image_bytes):
        """
        Preprocess image from bytes (for API upload)
        """
        try:
            # Convert bytes to PIL Image
            img = Image.open(image_bytes)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize(self.img_size)
            
            # Convert to array and normalize
            img_array = img_to_array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image from bytes: {str(e)}")
            return None
    
    def save_preprocessed_data(self, X, y, save_path):
        """
        Save preprocessed data to disk
        """
        try:
            np.savez_compressed(save_path, X=X, y=y)
            logger.info(f"Preprocessed data saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
    
    def load_preprocessed_data(self, load_path):
        """
        Load preprocessed data from disk
        """
        try:
            data = np.load(load_path)
            return data['X'], data['y']
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {str(e)}")
            return None, None
    
    def analyze_dataset(self, data_dir):
        """
        Analyze dataset statistics
        """
        stats = {}
        total_images = 0
        
        for class_name in self.classes:
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(image_files)
                stats[class_name] = count
                total_images += count
        
        stats['total_images'] = total_images
        stats['num_classes'] = len(self.classes)
        
        # Calculate imbalance ratio
        if len(stats) > 1:
            counts = [v for k, v in stats.items() if k not in ['total_images', 'num_classes']]
            if counts:
                stats['imbalance_ratio'] = max(counts) / min(counts)
        
        logger.info(f"Dataset analysis: {stats}")
        return stats
    
    def prepare_generators(self, data_dir=None, existing_data_dir=None, validation_split=0.2):
        """
        Prepare data generators for retraining
        Combines new data with existing data if available
        """
        if data_dir is None:
            raise ValueError("Data directory must be provided")
        
        # First, discover classes from the directory
        self.load_classes_from_directory(data_dir)
        
        # If existing data directory is provided, combine datasets
        combined_data_dir = data_dir
        if existing_data_dir and os.path.exists(existing_data_dir):
            logger.info("Combining new data with existing data for retraining")
            combined_data_dir = self._combine_datasets(new_data_dir=data_dir, existing_data_dir=existing_data_dir)
            # Reload classes from combined dataset
            self.load_classes_from_directory(combined_data_dir)
        
        # Create data generators for retraining
        train_gen, val_gen = self.create_data_generators(combined_data_dir, validation_split=validation_split)
        
        # Return generators and class names for model training
        return train_gen, val_gen, self.classes
    
    def _combine_datasets(self, new_data_dir, existing_data_dir):
        """
        Combine new training data with existing data
        Creates a temporary combined dataset directory
        """
        import shutil
        import tempfile
        
        # Create temporary directory for combined data
        combined_dir = os.path.join(tempfile.gettempdir(), "combined_training_data")
        if os.path.exists(combined_dir):
            shutil.rmtree(combined_dir)
        os.makedirs(combined_dir)
        
        # Copy existing data
        if os.path.exists(existing_data_dir):
            for class_name in os.listdir(existing_data_dir):
                src_class_dir = os.path.join(existing_data_dir, class_name)
                if os.path.isdir(src_class_dir):
                    dst_class_dir = os.path.join(combined_dir, class_name)
                    shutil.copytree(src_class_dir, dst_class_dir)
        
        # Add new data
        if os.path.exists(new_data_dir):
            for class_name in os.listdir(new_data_dir):
                src_class_dir = os.path.join(new_data_dir, class_name)
                if os.path.isdir(src_class_dir):
                    dst_class_dir = os.path.join(combined_dir, class_name)
                    
                    # If class already exists, add images to existing directory
                    if os.path.exists(dst_class_dir):
                        for file_name in os.listdir(src_class_dir):
                            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                src_file = os.path.join(src_class_dir, file_name)
                                dst_file = os.path.join(dst_class_dir, f"new_{file_name}")
                                shutil.copy2(src_file, dst_file)
                    else:
                        # New class, copy entire directory
                        shutil.copytree(src_class_dir, dst_class_dir)
        
        logger.info(f"Combined dataset created at: {combined_dir}")
        return combined_dir

    def prepare_retraining_data(self, new_data_dir, existing_data_dir=None):
        """
        Prepare data for retraining by combining new and existing data
        """
        return self.prepare_generators(new_data_dir, existing_data_dir)
    
    def save_class_mappings(self, save_dir):
        """
        Save class mappings to JSON files
        """
        try:
            # Save class mappings
            with open(os.path.join(save_dir, 'class_to_idx.json'), 'w') as f:
                json.dump(self.class_to_idx, f, indent=4)
            
            with open(os.path.join(save_dir, 'idx_to_class.json'), 'w') as f:
                json.dump(self.idx_to_class, f, indent=4)
            
            with open(os.path.join(save_dir, 'classes.json'), 'w') as f:
                json.dump(self.classes, f, indent=4)
            
            logger.info(f"Class mappings saved to {save_dir}")
            
        except Exception as e:
            logger.error(f"Error saving class mappings: {str(e)}")
    
    def load_class_mappings(self, load_dir):
        """
        Load class mappings from JSON files
        """
        try:
            with open(os.path.join(load_dir, 'class_to_idx.json'), 'r') as f:
                self.class_to_idx = json.load(f)
            
            with open(os.path.join(load_dir, 'idx_to_class.json'), 'r') as f:
                # Convert string keys back to integers
                self.idx_to_class = {int(k): v for k, v in json.load(f).items()}
            
            with open(os.path.join(load_dir, 'classes.json'), 'r') as f:
                self.classes = json.load(f)
            
            logger.info(f"Class mappings loaded from {load_dir}")
            
        except Exception as e:
            logger.error(f"Error loading class mappings: {str(e)}")

def create_test_train_split(data_dir, test_size=0.2, random_state=42):
    """
    Create train/test split while maintaining directory structure
    """
    # This function would implement file-based train/test splitting
    # For now, we'll use the data generator approach
    pass

if __name__ == "__main__":
    # Example usage
    preprocessor = FoodDataPreprocessor()
    
    # Load classes
    data_dir = "../data"
    if os.path.exists(data_dir):
        classes = preprocessor.load_classes_from_directory(data_dir)
        
        # Create data generators
        train_gen, val_gen = preprocessor.create_data_generators(data_dir)
        
        # Analyze dataset
        stats = preprocessor.analyze_dataset(data_dir)
        
        # Save class mappings
        preprocessor.save_class_mappings("../models")
    else:
        print(f"Data directory {data_dir} not found")
