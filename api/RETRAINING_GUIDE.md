"""
Retraining Data Structure Documentation
=====================================

For proper retraining, uploaded files should follow this directory structure:

## Option 1: Category-based Upload (Recommended)

When uploading training data via API, specify the category parameter:

POST /upload/training-data?category=Bread

- Files: [bread1.jpg, bread2.jpg, ...]

POST /upload/training-data?category=Dessert

- Files: [cake1.jpg, cookie1.jpg, ...]

This creates:
retrain_data/
├── Bread/
│ ├── bread1.jpg
│ ├── bread2.jpg
│ └── ...
├── Dessert/
│ ├── cake1.jpg
│ ├── cookie1.jpg
│ └── ...
└── ...

## Option 2: Bulk Directory Upload

Upload a zip file with this structure:
training_data.zip
├── Bread/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── Dairy product/
│ ├── image1.jpg
│ └── ...
└── ...

Supported Formats:

- Images: .jpg, .jpeg, .png
- Minimum 10 images per class recommended
- Images will be automatically resized to 224x224

Class Handling:

- New classes will be added to the model
- Existing classes will be updated with new data
- Class mappings are automatically managed
- Original data is preserved and combined with new data

Training Process:

1. Upload images with category labels
2. Call /retrain endpoint
3. System automatically:
   - Discovers all classes in retrain_data/
   - Combines with original training data
   - Updates class mappings
   - Trains new model version
   - Activates new model upon completion

Quality Guidelines:

- High-quality, well-lit images
- Clear subject matter
- Variety in angles, lighting, backgrounds
- Minimum 224x224 resolution (will be resized)
- Representative samples of each food category
