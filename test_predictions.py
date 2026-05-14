#!/usr/bin/env python3
"""
Test script to verify model predictions are working correctly
"""

import numpy as np
from keras.models import load_model
import os
from pathlib import Path

def test_corrected_samples():
    """Test all NPY samples from sample_images_corrected/"""

    print("="*80)
    print("Testing Corrected NPY Samples")
    print("="*80)
    print()

    # Load model
    print("Loading model...")
    model = load_model('final_novel_attention_model.keras')
    print("✅ Model loaded successfully")
    print()

    # Test corrected samples
    corrected_dir = Path('sample_images_corrected')
    npy_files = sorted(corrected_dir.glob('*.npy'))

    print(f"Found {len(npy_files)} NPY samples")
    print()

    correct_count = 0
    total_count = 0

    for npy_file in npy_files:
        # Load preprocessed image
        img = np.load(npy_file)
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img_batch, verbose=0)
        pred_val = float(prediction[0][0])

        # Determine expected result
        is_malignant = 'malignant' in npy_file.name
        expected = 'CANCER' if is_malignant else 'NO CANCER'
        predicted = 'CANCER' if pred_val > 0.5 else 'NO CANCER'

        # Check if correct
        is_correct = (expected == predicted)
        status = '✅' if is_correct else '❌'

        if is_correct:
            correct_count += 1
        total_count += 1

        # Print result
        print(f'{status} {npy_file.name:20s}: {pred_val:.6f} → {predicted:12s} (expected: {expected})')

    print()
    print("="*80)
    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}% accuracy)")
    print("="*80)
    print()

    if correct_count == total_count:
        print("🎉 SUCCESS! All predictions are correct!")
        return True
    else:
        print(f"⚠️ WARNING: {total_count - correct_count} predictions were incorrect")
        return False


def test_png_samples():
    """Test PNG samples (expected to be less accurate)"""

    print()
    print("="*80)
    print("Testing Original PNG Samples (Less Accurate)")
    print("="*80)
    print()

    from PIL import Image
    import cv2
    import tensorflow as tf

    # Load model
    model = load_model('final_novel_attention_model.keras')

    # PNG preprocessing function (from app.py)
    def preprocess_png(image):
        img_array = np.array(image)

        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array

        img_gray = img_gray.astype('float32')
        img_hu_approx = (img_gray / 255.0) * 1400.0 - 1000.0
        img_3channel = np.stack([img_hu_approx] * 3, axis=-1)
        img_resized = tf.image.resize(img_3channel, (96, 96))
        img_normalized = img_resized.numpy().astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        return img_batch

    # Test PNG samples
    png_dir = Path('sample_images')
    png_files = sorted([f for f in png_dir.glob('*.png') if f.name != 'README.txt'])

    print(f"Found {len(png_files)} PNG samples")
    print()

    correct_count = 0
    total_count = 0

    for png_file in png_files:
        # Load and preprocess
        image = Image.open(png_file)
        img_batch = preprocess_png(image)

        # Predict
        prediction = model.predict(img_batch, verbose=0)
        pred_val = float(prediction[0][0])

        # Determine expected result
        is_malignant = 'malignant' in png_file.name
        expected = 'CANCER' if is_malignant else 'NO CANCER'
        predicted = 'CANCER' if pred_val > 0.5 else 'NO CANCER'

        # Check if correct
        is_correct = (expected == predicted)
        status = '✅' if is_correct else '❌'

        if is_correct:
            correct_count += 1
        total_count += 1

        # Print result
        print(f'{status} {png_file.name:20s}: {pred_val:.6f} → {predicted:12s} (expected: {expected})')

    print()
    print("="*80)
    print(f"Results: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}% accuracy)")
    print("="*80)
    print()
    print("Note: PNG files lose HU information, so lower accuracy is expected.")
    print("      Use NPY files for production inference.")
    print()


if __name__ == "__main__":
    print()
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "MODEL PREDICTION TEST SUITE" + " "*31 + "║")
    print("╚" + "="*78 + "╝")
    print()

    # Test corrected samples (should be 100% accurate)
    npy_success = test_corrected_samples()

    # Test PNG samples (will be less accurate)
    test_png_samples()

    # Final summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    if npy_success:
        print("✅ NPY samples: PASSED (100% accuracy)")
        print("✅ Model is working correctly with proper preprocessing")
    else:
        print("❌ NPY samples: FAILED")
        print("⚠️  Check preprocessing pipeline")

    print()
    print("Recommendation:")
    print("  • Use NPY files from sample_images_corrected/ for accurate predictions")
    print("  • PNG files are for visualization only")
    print()
    print("To run the app: ./run_app.sh or streamlit run app.py")
    print("="*80)
    print()
