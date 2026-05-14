CT Scan Sample Images (Corrected - HU Preserving)
================================================================================

This folder contains lung nodule CT scan samples with proper HU preservation.

Files:
--------------------------------------------------------------------------------
*.npy files: Preprocessed images (96x96x3) ready for model.predict()
             These are already preprocessed identically to training data.

*.png files: Visualization images (for human viewing only)
             These use min-max normalization for better contrast.
             DO NOT use these for model predictions!

metadata.json: Contains HU ranges and ground truth labels

Expected Predictions:
--------------------------------------------------------------------------------
benign_*.npy:     Label 0 → Model should predict < 0.5 (NO CANCER)
malignant_*.npy:  Label 1 → Model should predict > 0.5 (CANCER)

Usage Example (Python):
--------------------------------------------------------------------------------
import numpy as np
from keras.models import load_model

# Load model
model = load_model('final_novel_attention_model.keras')

# Load preprocessed image
img = np.load('sample_images_corrected/malignant_001.npy')

# Add batch dimension
img_batch = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img_batch)
print(f'Prediction: {prediction[0][0]:.4f}')

