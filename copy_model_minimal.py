#!/usr/bin/env python3
"""
Copy only essential model files for nnU-Net inference.
Avoids copying huge prediction folders and unnecessary files.
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime


def copy_minimal_model_files(source_model_dir, dest_dir, fold=1):
    """Copy only essential files for inference.
    
    Args:
        source_model_dir: Path to trained model directory 
        dest_dir: Destination directory for minimal model files
        fold: Which fold to copy (default: 1, the best performing fold)
    """
    source_path = Path(source_model_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source model directory not found: {source_path}")
    
    # Create destination
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Essential files to copy (only what's needed for inference)
    essential_files = [
        "plans.json",              # Model plans
        "dataset.json",            # Dataset info  
        f"fold_{fold}/checkpoint_final.pth",  # Trained model weights
        f"fold_{fold}/checkpoint_best.pth"    # Best checkpoint (if exists)
    ]
    
    # Optional files (copy if they exist)
    optional_files = [
        "postprocessing.pkl",      # Postprocessing parameters
    ]
    
    copied_files = []
    skipped_files = []
    
    print(f"📂 Copying minimal model files...")
    print(f"Source: {source_path}")
    print(f"Dest:   {dest_path}")
    print("-" * 50)
    
    # Copy essential files
    for file_path in essential_files:
        src_file = source_path / file_path
        dst_file = dest_path / file_path
        
        if src_file.exists():
            # Create parent directory if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(src_file, dst_file)
            file_size = dst_file.stat().st_size / (1024*1024)  # MB
            copied_files.append(file_path)
            print(f"✅ {file_path:<30} ({file_size:.1f} MB)")
        else:
            skipped_files.append(file_path)
            print(f"❌ {file_path:<30} (not found)")
    
    # Copy optional files
    for file_path in optional_files:
        src_file = source_path / file_path
        dst_file = dest_path / file_path
        
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            file_size = dst_file.stat().st_size / (1024*1024)  # MB
            copied_files.append(file_path)
            print(f"📄 {file_path:<30} ({file_size:.1f} MB)")
    
    print("-" * 50)
    print(f"✅ Copied {len(copied_files)} files")
    if skipped_files:
        print(f"⚠️  Skipped {len(skipped_files)} missing files")
    
    # Calculate total size
    total_size = sum((dest_path / f).stat().st_size for f in copied_files if (dest_path / f).exists()) / (1024*1024)
    print(f"📊 Total size: {total_size:.1f} MB")
    
    return copied_files, skipped_files

def create_model_config(dest_dir, fold, stage_name):
    """Create a config file with model metadata."""
    # Validation performance for each fold (from 5-fold CV results)
    
    config = {
        'model_info': {
            'fold': fold,
            'stage': stage_name,
            'architecture': 'nnUNet ResEnc (Residual Encoder)',
            'configuration': '3d_cascade_fullres' if stage_name == 'cascade' else '3d_lowres',
            'planner': 'nnUNetResEncUNetMPlans'
        },
        'deployment': {
            'copied_date': datetime.now().isoformat(),
        },
        'labels': {
            '0': 'background',
            '1': 'patellar_cartilage',
            '2': 'femoral_cartilage',
            '3': 'medial_tibial_cartilage',
            '4': 'lateral_tibial_cartilage',
            '5': 'medial_meniscus',
            '6': 'lateral_meniscus',
            '7': 'femur_bone',
            '8': 'tibia_bone',
            '9': 'patella_bone'
        }
    }
    
    config_file = dest_dir / 'model_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"  📄 Created model_config.json")
    return config_file

def copy_lowres_for_cascade(fold=1):
    """Copy lowres model for cascade compatibility."""
    # Source: your trained lowres (standard trainer)
    nnunet_results = os.environ.get('nnUNet_results', '')
    if not nnunet_results:
        # Use the default path
        nnunet_results = "/dataNAS/people/aagatti/projects/knee_pipeline_nnunet/nnunet_data/nnUNet_results"
    
    source_lowres = Path(nnunet_results) / "Dataset500_KneeMRI" / "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres"
    
    # Destination: standard naming for cascade compatibility
    dest_lowres = Path("./huggingface/models/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres")
    
    if not source_lowres.exists():
        print(f"⚠️  Lowres model not found at: {source_lowres}")
        return False
    
    print(f"🔄 Copying lowres model (fold {fold}) for cascade compatibility...")
    
    # Copy using the same function
    try:
        copied, skipped = copy_minimal_model_files(source_lowres, dest_lowres, fold=fold)
        
        # Create config file for lowres
        create_model_config(dest_lowres, fold, 'lowres')
        
        print(f"✅ Lowres model copied: {dest_lowres}")
        return True
    except Exception as e:
        print(f"❌ Error copying lowres: {e}")
        return False

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy minimal nnU-Net model files for inference")
    parser.add_argument("--source", 
                       help="Source model directory (default: your cascade fullres model)")
    parser.add_argument("--dest", 
                       help="Destination directory (default: ./huggingface/models/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres)")
    parser.add_argument("--fold", type=int, default=1, choices=[0,1,2,3,4],
                       help="Which fold to copy (default: 1, the best performing fold)")
    parser.add_argument("--setup_lowres", action="store_true",
                       help="Also setup lowres model for cascade compatibility")
    
    args = parser.parse_args()
    
    # Set defaults based on your actual structure
    if args.source is None:
        nnunet_results = os.environ.get('nnUNet_results', '/dataNAS/people/aagatti/projects/knee_pipeline_nnunet/nnunet_data/nnUNet_results')
        args.source = f"{nnunet_results}/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres"
    
    if args.dest is None:
        args.dest = "./huggingface/models/Dataset500_KneeMRI/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_cascade_fullres"
    
    try:
        # Copy cascade fullres model
        copied, skipped = copy_minimal_model_files(args.source, args.dest, fold=args.fold)
        
        # Create config file for cascade
        config_file = create_model_config(Path(args.dest), args.fold, 'cascade')
        print(f"  ✅ Model config created with fold {args.fold} info")
        
        # Setup lowres if requested
        if args.setup_lowres:
            print()
            lowres_success = copy_lowres_for_cascade(fold=args.fold)
            if not lowres_success:
                print("⚠️  Lowres setup failed, but cascade model copied successfully.")
        
        if skipped:
            print(f"\n⚠️  Some essential files were missing:")
            for file in skipped:
                print(f"   - {file}")
            print(f"\nInference may not work properly. Check your source directory.")
            return 1
        else:
            print(f"\n🎉 Minimal model copy complete!")
            print(f"📁 Cascade model ready at: {args.dest}")
            if args.setup_lowres:
                print(f"📁 Lowres model copied for cascade compatibility")
            print(f"\nNext step: python test_inference.py")
            return 0
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())