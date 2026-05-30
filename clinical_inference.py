"""
Clinical Inference Pipeline - Real-world CT Scan Processing
Accepts any CT scan format (DICOM, PNG, TIFF, JPG) and performs nodule classification

This script mimics real clinical workflow:
1. Load CT scan (any format)
2. Preprocess and normalize
3. Extract lung regions (optional)
4. Detect potential nodules (optional sliding window)
5. Classify each nodule
6. Generate clinical report with Grad-CAM visualizations

Author: MTech Research Project
"""

import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import load_model, Model
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json

# Try to import DICOM support (optional)
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("⚠ Warning: pydicom not installed. DICOM support disabled.")
    print("  Install with: pip install pydicom")

# Configuration
CONFIG = {
    'target_size': (96, 96),
    'model_path': 'final_novel_attention_model.keras',
    'confidence_threshold': 0.5,
    'window_size': 96,
    'stride': 32,  # For sliding window detection
}


class CTScanProcessor:
    """Processes CT scans from various formats"""

    def __init__(self):
        self.supported_formats = ['.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']

    def load_scan(self, file_path):
        """
        Load CT scan from any supported format

        Args:
            file_path: Path to CT scan file

        Returns:
            image: numpy array (H, W) or (H, W, C)
            metadata: dict with scan information
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {file_ext}. Supported: {self.supported_formats}")

        metadata = {
            'file_path': file_path,
            'format': file_ext,
            'load_time': datetime.now().isoformat()
        }

        # DICOM format
        if file_ext == '.dcm':
            if not DICOM_AVAILABLE:
                raise ImportError("pydicom required for DICOM files. Install with: pip install pydicom")

            dcm = pydicom.dcmread(file_path)
            image = dcm.pixel_array.astype(np.float32)

            # Extract metadata
            metadata.update({
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'slice_thickness': getattr(dcm, 'SliceThickness', 'Unknown'),
                'pixel_spacing': getattr(dcm, 'PixelSpacing', 'Unknown'),
            })

            # Apply DICOM windowing if available
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                window_center = float(dcm.WindowCenter)
                window_width = float(dcm.WindowWidth)
                image = self.apply_windowing(image, window_center, window_width)

        # NumPy array
        elif file_ext == '.npy':
            image = np.load(file_path).astype(np.float32)

        # Standard image formats (PNG, JPG, TIFF)
        else:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Try with PIL
                image = np.array(Image.open(file_path).convert('L')).astype(np.float32)

        metadata['original_shape'] = image.shape
        metadata['original_dtype'] = str(image.dtype)

        return image, metadata

    def apply_windowing(self, image, window_center, window_width):
        """Apply CT windowing (Hounsfield Units)"""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        image = np.clip(image, img_min, img_max)
        return image

    def normalize_ct_scan(self, image):
        """Normalize CT scan to [0, 1] range"""
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        return image.astype(np.float32)

    def extract_lung_region(self, image):
        """
        Simple lung region extraction using thresholding
        (For production, use U-Net or similar segmentation)
        """
        # Normalize
        norm_img = self.normalize_ct_scan(image.copy())
        norm_img_uint8 = (norm_img * 255).astype(np.uint8)

        # Apply threshold to isolate lungs (lungs are darker in CT)
        _, binary = cv2.threshold(norm_img_uint8, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return image  # Return original if no lungs detected

        # Get largest contours (likely lungs)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        # Create mask
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Apply mask
        lung_region = cv2.bitwise_and(norm_img_uint8, norm_img_uint8, mask=mask)

        return lung_region

    def sliding_window_detection(self, image, window_size=96, stride=32):
        """
        Sliding window to detect potential nodules

        Returns:
            patches: list of (patch, x, y) tuples
        """
        h, w = image.shape[:2]
        patches = []

        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                patch = image[y:y+window_size, x:x+window_size]

                # Only keep patches with sufficient variation (likely contain nodules)
                if patch.std() > 10:  # Threshold can be tuned
                    patches.append((patch, x, y))

        return patches

    def preprocess_for_model(self, patch, target_size=(96, 96)):
        """
        Preprocess patch for model input

        Args:
            patch: numpy array (H, W) or (H, W, C)

        Returns:
            preprocessed: (1, target_size[0], target_size[1], 3)
        """
        # Ensure grayscale
        if len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Resize
        patch_resized = cv2.resize(patch, target_size)

        # Normalize
        patch_normalized = self.normalize_ct_scan(patch_resized)

        # Convert to RGB (3 channels)
        patch_rgb = np.stack([patch_normalized] * 3, axis=-1)

        # Add batch dimension
        patch_batch = np.expand_dims(patch_rgb, axis=0)

        return patch_batch.astype(np.float32)


class ClinicalInference:
    """Clinical inference with Grad-CAM visualization"""

    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.processor = CTScanProcessor()

    def predict_single_patch(self, patch):
        """Predict on a single patch"""
        preprocessed = self.processor.preprocess_for_model(patch)
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]
        return prediction

    def generate_gradcam(self, patch, last_conv_layer_name=None):
        """Generate Grad-CAM heatmap"""
        # Find last conv layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break

        if last_conv_layer_name is None:
            print("⚠ Warning: No convolutional layer found for Grad-CAM")
            return None

        # Preprocess
        img_array = self.processor.preprocess_for_model(patch)

        # Create gradient model
        grad_model = Model(
            [self.model.input],
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Generate heatmap
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

        return heatmap.numpy()

    def process_full_scan(self, scan_path, output_dir='clinical_output', use_sliding_window=False):
        """
        Process a full CT scan

        Args:
            scan_path: Path to CT scan file
            output_dir: Directory to save results
            use_sliding_window: If True, detect nodules using sliding window
        """
        print("\n" + "="*80)
        print("CLINICAL INFERENCE PIPELINE")
        print("="*80)

        os.makedirs(output_dir, exist_ok=True)

        # Load scan
        print(f"\n[1/5] Loading scan: {scan_path}")
        image, metadata = self.processor.load_scan(scan_path)
        print(f"  ✓ Loaded: {image.shape}, dtype: {image.dtype}")

        # Normalize
        print("\n[2/5] Normalizing...")
        image_normalized = self.processor.normalize_ct_scan(image)

        results = {
            'scan_info': metadata,
            'detections': [],
            'summary': {}
        }

        if use_sliding_window:
            # Sliding window detection
            print("\n[3/5] Detecting potential nodules (sliding window)...")
            patches = self.processor.sliding_window_detection(
                image_normalized,
                window_size=CONFIG['window_size'],
                stride=CONFIG['stride']
            )
            print(f"  ✓ Found {len(patches)} candidate regions")

            print("\n[4/5] Classifying detected regions...")
            detections = []

            for i, (patch, x, y) in enumerate(patches):
                prediction = self.predict_single_patch(patch)

                if prediction > CONFIG['confidence_threshold']:
                    detections.append({
                        'patch_id': i,
                        'position': (x, y),
                        'confidence': float(prediction),
                        'classification': 'Malignant' if prediction > 0.5 else 'Benign'
                    })

            results['detections'] = detections
            print(f"  ✓ Detected {len(detections)} suspicious regions")

        else:
            # Process entire scan as single patch
            print("\n[3/5] Preprocessing scan...")
            patch = cv2.resize(image_normalized, CONFIG['target_size'])

            print("\n[4/5] Running classification...")
            prediction = self.predict_single_patch(patch)

            results['detections'] = [{
                'patch_id': 0,
                'position': (0, 0),
                'confidence': float(prediction),
                'classification': 'Malignant' if prediction > 0.5 else 'Benign'
            }]

        # Generate report
        print("\n[5/5] Generating clinical report...")
        self.generate_report(image, results, output_dir)

        print("\n" + "="*80)
        print("✓ INFERENCE COMPLETE")
        print("="*80)
        print(f"Results saved to: {output_dir}/")

        return results

    def generate_report(self, original_image, results, output_dir):
        """Generate clinical report with visualizations"""

        # Save results JSON
        with open(f'{output_dir}/clinical_report.json', 'w') as f:
            json.dump(results, indent=2, fp=f)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Original scan
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title('Original CT Scan', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Detection overlay
        detection_img = np.stack([original_image] * 3, axis=-1)
        detection_img = (detection_img - detection_img.min()) / (detection_img.max() - detection_img.min() + 1e-10)
        detection_img = (detection_img * 255).astype(np.uint8)

        for detection in results['detections']:
            x, y = detection['position']
            confidence = detection['confidence']
            color = (255, 0, 0) if confidence > 0.5 else (0, 255, 0)

            cv2.rectangle(detection_img, (x, y), (x + CONFIG['window_size'], y + CONFIG['window_size']),
                         color, 2)
            cv2.putText(detection_img, f"{confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[1].imshow(detection_img)
        axes[1].set_title('Detection Results', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/clinical_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generate text report
        report_text = f"""
CLINICAL INFERENCE REPORT
{'='*80}

Scan Information:
- File: {results['scan_info']['file_path']}
- Format: {results['scan_info']['format']}
- Original Shape: {results['scan_info']['original_shape']}
- Analysis Time: {results['scan_info']['load_time']}

Detection Results:
- Total Detections: {len(results['detections'])}
- Malignant (>0.5): {sum(1 for d in results['detections'] if d['confidence'] > 0.5)}
- Benign (≤0.5): {sum(1 for d in results['detections'] if d['confidence'] <= 0.5)}

Detailed Findings:
"""
        for i, detection in enumerate(results['detections'], 1):
            report_text += f"""
Detection #{i}:
  Position: {detection['position']}
  Confidence: {detection['confidence']:.4f}
  Classification: {detection['classification']}
  Recommendation: {'Further evaluation recommended' if detection['confidence'] > 0.7 else 'Monitor'}
"""

        report_text += f"""
{'='*80}

IMPORTANT NOTICE:
This is a research prototype. All findings must be verified by a qualified
radiologist before making any clinical decisions.

Generated by: MTech Lung Nodule Classification System
Model: MobileNetV2 + Dual Attention
"""

        with open(f'{output_dir}/clinical_report.txt', 'w') as f:
            f.write(report_text)

        print(f"  ✓ Report saved: {output_dir}/clinical_report.txt")
        print(f"  ✓ Visualization saved: {output_dir}/clinical_visualization.png")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Clinical CT Scan Inference')

    parser.add_argument('scan_path', type=str, help='Path to CT scan file')
    parser.add_argument('--model', type=str, default=CONFIG['model_path'],
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='clinical_output',
                       help='Output directory for results')
    parser.add_argument('--sliding-window', action='store_true',
                       help='Use sliding window detection (slower but more thorough)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for malignancy')

    args = parser.parse_args()

    # Update config
    CONFIG['model_path'] = args.model
    CONFIG['confidence_threshold'] = args.threshold

    # Check if scan exists
    if not os.path.exists(args.scan_path):
        print(f"❌ Error: File not found: {args.scan_path}")
        sys.exit(1)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        print(f"   Please train the model first by running: jupyter notebook main.ipynb")
        sys.exit(1)

    # Run inference
    try:
        inference = ClinicalInference(args.model)
        results = inference.process_full_scan(
            args.scan_path,
            output_dir=args.output,
            use_sliding_window=args.sliding_window
        )

        print("\n✓ Analysis complete! Check the output directory for results.")

    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
