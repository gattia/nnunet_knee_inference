#!/usr/bin/env python3
"""
Simple nnU-Net inference for knee MRI segmentation using SimpleITK.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import tempfile
import shutil
import subprocess

class KneeSegmentationInference:
    """Simple nnU-Net inference class."""
    
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "huggingface" / "models" / "Dataset500_KneeMRI" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres"
        
        # No label remapping - output sequential nnU-Net labels (0-9) as trained
        
        self._check_setup()
    
    def _check_setup(self):
        """Check if nnU-Net and model are available."""
        try:
            import nnunetv2
            print("✅ nnU-Net v2 found")
        except ImportError:
            raise ImportError("nnunetv2 not found. Install with: pip install nnunetv2")
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
    
    def predict(self, input_path, output_path=None):
        """Run inference on a single image.
        
        Args:
            input_path: Path to input image (.nii.gz or .nrrd)
            output_path: Path to save output (optional)
            
        Returns:
            SimpleITK image with segmentation
        """
        input_path = Path(input_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = Path(temp_dir) / "input"
            temp_output = Path(temp_dir) / "output" 
            temp_input.mkdir()
            temp_output.mkdir()
            
            # Prepare input for nnU-Net (needs _0000.nii.gz format)
            case_name = input_path.stem.replace('.nii', '').replace('.nrrd', '')
            nnunet_input = temp_input / f"{case_name}_0000.nii.gz"
            
            # Convert to NIfTI if needed
            img = sitk.ReadImage(str(input_path))
            sitk.WriteImage(img, str(nnunet_input))
            
            # Set temporary nnU-Net environment to point to our model directory
            env = os.environ.copy()
            env['nnUNet_results'] = str(self.model_dir.parent.parent)  # Point to huggingface/models/
            
            # For cascade, we need to run lowres first, then fullres (like the official script)
            temp_lowres = Path(temp_dir) / "lowres_predictions"
            temp_lowres.mkdir()
            
            # Step 1: Run 3d_lowres prediction first
            print("🔄 Running 3d_lowres prediction...")
            cmd_lowres = [
                'nnUNetv2_predict',
                '-i', str(temp_input),
                '-o', str(temp_lowres),
                '-d', 'Dataset500_KneeMRI',
                '-tr', 'nnUNetTrainer',
                '-p', 'nnUNetResEncUNetMPlans',
                '-c', '3d_lowres',
                '-f', '0',
                '-chk', 'checkpoint_best.pth',
                '--disable_tta',
                '-device', 'cuda'
            ]
            
            result = subprocess.run(cmd_lowres, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"3d_lowres prediction failed: {result.stderr}")
            
            # Step 2: Run 3d_cascade_fullres with lowres predictions
            print("🔄 Running 3d_cascade_fullres prediction...")
            cmd_fullres = [
                'nnUNetv2_predict',
                '-i', str(temp_input),
                '-o', str(temp_output),
                '-d', 'Dataset500_KneeMRI',
                '-tr', 'nnUNetTrainer',
                '-p', 'nnUNetResEncUNetMPlans',
                '-c', '3d_cascade_fullres',
                '-f', '0',
                '-chk', 'checkpoint_best.pth',
                '-prev_stage_predictions', str(temp_lowres),
                '--disable_tta',
                '-device', 'cuda'
            ]
            
            result = subprocess.run(cmd_fullres, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"3d_cascade_fullres prediction failed: {result.stderr}")
            
            # Load result
            result_file = next(temp_output.glob("*.nii.gz"))
            seg_img = sitk.ReadImage(str(result_file))
            
            # Keep original nnU-Net sequential labels (0-9) - no remapping
            # Other libraries can handle remapping as needed
            output_img = seg_img
            output_img.CopyInformation(img)  # Copy spacing, origin, direction from input
            
            # Save if requested
            if output_path:
                sitk.WriteImage(output_img, str(output_path))
                print(f"✅ Segmentation saved: {output_path}")
            
            return output_img


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="nnU-Net Knee MRI Segmentation")
    parser.add_argument("--input", required=True, help="Input image (.nii.gz or .nrrd)")
    parser.add_argument("--output", required=True, help="Output segmentation")
    parser.add_argument("--model_dir", help="Model directory path")
    
    args = parser.parse_args()
    
    try:
        inference = KneeSegmentationInference(args.model_dir)
        result = inference.predict(args.input, args.output)
        
        # Show label info
        seg_array = sitk.GetArrayFromImage(result)
        unique_labels = np.unique(seg_array)
        print(f"✅ Complete! Found labels: {sorted(unique_labels)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())