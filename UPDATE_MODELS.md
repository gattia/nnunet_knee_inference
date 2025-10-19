# Updating Models After Training

## Quick Workflow

### 1. Copy New Model to Inference Package

```bash
cd /dataNAS/people/aagatti/projects/knee_pipeline_nnunet/nnunet_knee_inference

# Copy both cascade and lowres (use best fold)
python copy_model_minimal.py --fold 1 --setup_lowres
```

This copies model files (~1.6 GB) from `nnUNet_results` to `./huggingface/models/`

### 2. Test Locally

```bash
python test_inference.py
```

### 3. Update HuggingFace

```bash
# Login once (if needed)
huggingface-cli login

# Upload new models
huggingface-cli upload aagatti/nnunet_knee ./huggingface --repo-type model --commit-message "Update to fold X models"
```

**To delete old models:** Delete them directly from the HuggingFace web interface before uploading the new ones.

---

## Notes

- The script auto-creates `model_config.json` with fold info
- Inference scripts auto-detect which fold to use
- Both lowres and cascade should use the same fold