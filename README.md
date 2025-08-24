# nnU-Net Knee MRI Inference

Portable inference package for knee MRI segmentation using nnU-Net cascade architecture.

**Performance**: 0.906 mean Dice coefficient, ~80 seconds per volume

## Quick Start

### 1. Install Dependencies
```bash
pip install torch nnunetv2 SimpleITK numpy huggingface_hub
```

### 2. Download Models & Test Data
```bash
# Download models and test data from HuggingFace (one-time setup)
python download_models.py
```

### 3. Run Inference
```bash
# Command line
python scripts/inference.py --input image.nii.gz --output segmentation.nii.gz

# Python API
from scripts.inference import KneeSegmentationInference
inference = KneeSegmentationInference()
result = inference.predict("image.nii.gz", "seg.nii.gz")
```

### 4. Validate Setup (Optional)
```bash
python test_inference.py  # Runs full validation with test data
```

## Output Labels
- **0**: Background  
- **1**: Patellar cartilage | **2**: Femoral cartilage  
- **3**: Medial tibial cartilage | **4**: Lateral tibial cartilage  
- **5**: Medial meniscus | **6**: Lateral meniscus  
- **7**: Femur bone | **8**: Tibia bone | **9**: Patella bone

## Key Files
- `scripts/inference.py` - Main inference script and Python API
- `download_models.py` - Downloads models and test data from HuggingFace Hub  
- `test_inference.py` - Comprehensive validation with Dice metrics
- `huggingface/` - Model files, test data, and outputs (from HuggingFace)

## Architecture
Uses nnU-Net cascade: **3d_lowres** → **3d_cascade_fullres** for optimal accuracy.

## Requirements
- Python 3.10+
- PyTorch, nnunetv2, SimpleITK
- ~1GB disk space for models
- Supports .nii.gz and .nrrd formats

---
*Designed for easy integration into medical imaging pipelines.*