#!/usr/bin/env python3
"""
Download nnU-Net knee MRI models from HuggingFace Hub.
This script downloads the complete model repository into the huggingface/ folder.
"""

import argparse
from pathlib import Path
import sys

def download_models(repo_id="aagatti/nnunet_knee", local_dir="./huggingface"):
    """Download models and test data from HuggingFace Hub."""
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed. Install with:")
        print("   pip install huggingface_hub")
        return False
    
    print(f"📥 Downloading models and test data from {repo_id}...")
    print(f"📁 Target directory: {local_dir}")
    
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="model"
        )
        print("✅ Download completed successfully!")
        print(f"📂 Models are now available in: {local_dir}/models/")
        print(f"🧪 Test data available in: {local_dir}/test_data/")
        print("\n🚀 Ready to run inference:")
        print("   python scripts/inference.py --input image.nii.gz --output seg.nii.gz")
        print("\n🧪 Or test with sample data:")
        print("   python test_inference.py")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download nnU-Net knee MRI models from HuggingFace")
    parser.add_argument("--repo", default="aagatti/nnunet_knee",
                       help="HuggingFace repository ID (default: aagatti/nnunet_knee)")
    parser.add_argument("--dir", default="./huggingface",
                       help="Local directory to download to (default: ./huggingface)")
    
    args = parser.parse_args()
    
    success = download_models(args.repo, args.dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
