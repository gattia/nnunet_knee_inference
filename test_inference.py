#!/usr/bin/env python3
"""
Comprehensive test script for nnU-Net knee MRI inference.
Tests the complete pipeline and validates performance against ground truth.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import time

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

def calculate_dice_coefficient(pred, gt, label):
    """Calculate Dice coefficient for a specific label."""
    pred_binary = (pred == label).astype(np.float32)
    gt_binary = (gt == label).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    total = np.sum(pred_binary) + np.sum(gt_binary)
    
    if total == 0:
        return 1.0 if np.sum(pred_binary) == 0 else 0.0
    
    return 2.0 * intersection / total

def test_environment():
    """Test if environment is properly set up."""
    print("🔧 Testing Environment Setup")
    print("-" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.cuda.device_count()} GPU(s) available")
        else:
            print("⚠️  CUDA: Not available (will use CPU)")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import nnunetv2
        print(f"✅ nnU-Net v2: Available")
    except ImportError:
        print("❌ nnU-Net v2 not found")
        return False
    
    try:
        import SimpleITK as sitk
        print(f"✅ SimpleITK: {sitk.Version_VersionString()}")
    except ImportError:
        print("❌ SimpleITK not found")
        return False
    
    return True

def test_model_files():
    """Test if model files are present and valid."""
    print("\n🔍 Testing Model Files")
    print("-" * 40)
    
    base_dir = Path(__file__).parent
    model_dir = base_dir / "huggingface" / "models" / "Dataset500_KneeMRI" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres"
    
    # Try to read config file to determine which fold was deployed
    config_file = model_dir / "model_config.json"
    deployed_fold = None
    if config_file.exists():
        try:
            import json
            with open(config_file) as f:
                config = json.load(f)
            deployed_fold = config.get('model_info', {}).get('fold')
            print(f"📝 Config found: Fold {deployed_fold}")
        except Exception as e:
            print(f"⚠️  Could not read config: {e}")
    
    # Auto-detect which fold exists if config not found
    if deployed_fold is None:
        fold_dirs = list(model_dir.glob("fold_*"))
        if fold_dirs:
            deployed_fold = int(fold_dirs[0].name.split('_')[1])
            print(f"📁 Auto-detected fold: {deployed_fold}")
        else:
            print("❌ No fold directories found")
            return False
    
    # Check required files
    required_files = [
        "plans.json",
        "dataset.json", 
        f"fold_{deployed_fold}/checkpoint_best.pth"
    ]
    
    all_found = True
    for file_path in required_files:
        full_path = model_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024*1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} (missing)")
            all_found = False
    
    # Check lowres model
    lowres_dir = base_dir / "huggingface" / "models" / "Dataset500_KneeMRI" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres"
    lowres_config = lowres_dir / "model_config.json"
    if lowres_config.exists():
        try:
            import json
            with open(lowres_config) as f:
                lowres_conf = json.load(f)
            lowres_fold = lowres_conf.get('model_info', {}).get('fold')
            print(f"✅ Lowres model: fold {lowres_fold}")
        except:
            print("⚠️  Lowres config found but could not read")
    elif lowres_dir.exists():
        print(f"✅ Lowres model exists (no config)")
    else:
        print("⚠️  Lowres model not found")
    
    return all_found

def test_inference_pipeline():
    """Test the complete inference pipeline with performance validation."""
    print("\n🚀 Testing Inference Pipeline")
    print("-" * 40)
    
    # Test files - all in HuggingFace directory
    base_dir = Path(__file__).parent
    hf_test_dir = base_dir / "huggingface" / "test_data"
    
    test_image = hf_test_dir / "test_image.nii.gz"
    test_gt = hf_test_dir / "test_ground_truth.nii.gz"
    test_output = hf_test_dir / "test_prediction.nii.gz"
    
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        print("📥 Download test data with: python download_models.py")
        return False
    
    if not test_gt.exists():
        print(f"❌ Ground truth not found: {test_gt}")
        print("📥 Download test data with: python download_models.py")
        return False
    
    try:
        # Import inference class
        from scripts.inference import KneeSegmentationInference
        
        print(f"📊 Input image: {test_image.name}")
        
        # Load ground truth for comparison
        gt_img = sitk.ReadImage(str(test_gt))
        gt_array = sitk.GetArrayFromImage(gt_img)
        gt_labels = np.unique(gt_array)
        print(f"📊 Ground truth labels: {sorted(gt_labels)}")
        
        # Initialize inference
        print("🔄 Initializing inference...")
        inference = KneeSegmentationInference()
        
        # Run inference
        print("🔄 Running inference...")
        start_time = time.time()
        
        result_img = inference.predict(test_image, test_output)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"⏱️  Inference time: {inference_time:.1f} seconds")
        
        # Load prediction for validation
        pred_array = sitk.GetArrayFromImage(result_img)
        pred_labels = np.unique(pred_array)
        print(f"📊 Prediction labels: {sorted(pred_labels)}")
        
        # Calculate Dice coefficients
        print(f"\n📈 Dice Coefficient Analysis:")
        print("-" * 30)
        
        # Label names for reference (sequential nnU-Net labels 0-9)
        label_names = {
            0: "Background",
            1: "Patellar cartilage",
            2: "Femoral cartilage", 
            3: "Medial tibial cartilage",
            4: "Lateral tibial cartilage",
            5: "Medial meniscus",
            6: "Lateral meniscus",
            7: "Femur bone",
            8: "Tibia bone",
            9: "Patella bone"
        }
        
        dice_scores = {}
        all_labels = sorted(set(gt_labels) | set(pred_labels))
        
        for label in all_labels:
            if label == 0:  # Skip background
                continue
                
            dice = calculate_dice_coefficient(pred_array, gt_array, label)
            dice_scores[label] = dice
            
            label_name = label_names.get(label, f"Unknown_{label}")
            print(f"Label {label:2d} ({label_name:<20}): {dice:.3f}")
        
        # Calculate mean Dice for non-background labels
        if dice_scores:
            mean_dice = np.mean(list(dice_scores.values()))
            print(f"{'Mean Dice (non-bg)':<25}: {mean_dice:.3f}")
            
            # Check if performance is reasonable
            if mean_dice > 0.7:
                print("✅ Performance: Excellent (>0.7)")
            elif mean_dice > 0.5:
                print("⚠️  Performance: Acceptable (>0.5)")
            else:
                print("❌ Performance: Poor (<0.5)")
        
        # Summary
        print(f"\n📋 Test Summary:")
        print(f"  - Inference time: {inference_time:.1f}s")
        print(f"  - Mean Dice: {mean_dice:.3f}")
        print(f"  - Output saved: {test_output.name}")
        
        print(f"\n🎉 Inference test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 nnU-Net Knee MRI Inference Test Suite")
    print("=" * 50)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed!")
        return 1
    
    # Test model files
    if not test_model_files():
        print("\n❌ Model files test failed!")
        print("Run: python copy_model_minimal.py --setup_lowres")
        return 1
    
    # Test inference pipeline
    if not test_inference_pipeline():
        print("\n❌ Inference test failed!")
        return 1
    
    print(f"\n🎉 All tests passed! Inference pipeline is ready.")
    return 0

if __name__ == "__main__":
    exit(main())