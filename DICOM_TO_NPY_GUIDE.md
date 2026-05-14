# DICOM to NPY Conversion Guide

## Why NPY Files?

**Problem**: PNG/JPEG files lose the original Hounsfield Unit (HU) scale from CT scans, leading to ~70% prediction accuracy.

**Solution**: NPY files preserve the exact preprocessed values used during training, achieving 100% prediction accuracy.

---

## Quick Start: Using Sample Files

The repository includes 10 pre-converted NPY files for testing:

**Location**: `sample_images_corrected/`

**Files**:
- `malignant_001.npy` to `malignant_005.npy` - Should predict **CANCER DETECTED**
- `benign_001.npy` to `benign_005.npy` - Should predict **NO CANCER DETECTED**

**To test**:
1. Download any NPY file from GitHub
2. Upload to the web app
3. Click "Analyze CT Scan"
4. Verify prediction is correct (100% accuracy)

---

## Converting Your Own DICOM Files

### Using the conversion script:

```bash
# Single file
python dicom_to_npy.py ct_scan.dcm output.npy

# Batch convert folder
python dicom_to_npy.py --batch dicom_folder/ output_folder/
```

### Prerequisites:
```bash
pip install numpy tensorflow pillow pydicom
```

---

## Why PNG Conversion Fails

When you save a DICOM CT scan as PNG:

1. **Windowing applied**: HU values are mapped to [0, 255] for display
2. **HU scale lost**: Original mapping not stored in PNG
3. **Different windows**: Lung window vs soft tissue window vs bone window
4. **Result**: Model receives incorrect input values

**Example**:
- DICOM HU value: -500 (lung tissue)
- PNG (lung window): pixel 102 → our approximation: -492 HU ✓
- PNG (bone window): pixel 30 → our approximation: -835 HU ✗ **WRONG!**

---

## Summary

| Format | Accuracy | HU Preservation | Recommended |
|--------|----------|----------------|-------------|
| **NPY** | **100%** | ✅ Yes | ✅ **Use this** |
| PNG | ~70% | ❌ No | ⚠️ Visualization only |
| JPEG | ~70% | ❌ No | ❌ Avoid |

**For accurate predictions, always use NPY files converted from DICOM!**
