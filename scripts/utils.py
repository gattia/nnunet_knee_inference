"""
Simple utilities for nnU-Net inference.
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path

def get_image_info(image_path):
    """Get basic image information."""
    img = sitk.ReadImage(str(image_path))
    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    
    return {
        'size': size,
        'spacing': spacing,
        'origin': origin,
        'file_format': Path(image_path).suffix
    }

def print_label_summary(seg_img):
    """Print summary of segmentation labels."""
    seg_array = sitk.GetArrayFromImage(seg_img)
    unique_labels, counts = np.unique(seg_array, return_counts=True)
    total_voxels = seg_array.size
    
    print("\nSegmentation Summary:")
    print("-" * 30)
    for label, count in zip(unique_labels, counts):
        percentage = (count / total_voxels) * 100
        print(f"Label {int(label):2d}: {count:8,} voxels ({percentage:5.1f}%)")
    print("-" * 30)

def validate_input(input_path):
    """Basic input validation."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if input_path.suffix.lower() not in ['.gz', '.nrrd']:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Try to read the image
    try:
        img = sitk.ReadImage(str(input_path))
        return True
    except Exception as e:
        raise ValueError(f"Could not read image: {e}")

# Original label names for reference
LABEL_NAMES = {
    0: "background",
    1: "patellar_cartilage",
    2: "femoral_cartilage",
    3: "medial_tibial_cartilage", 
    4: "lateral_tibial_cartilage",
    5: "medial_meniscus",
    6: "lateral_meniscus",
    8: "femur_bone",         # Note: label 7 is skipped in original
    9: "tibia_bone",
    10: "patella_bone"
}