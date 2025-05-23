#!/usr/bin/env python3
"""
Script to upload an EnCodec model checkpoint to the Hugging Face Hub.
This script will:
1. Load the base EnCodec model
2. Load and apply your checkpoint
3. Save the model and processor locally
4. Push everything to the Hugging Face Hub

Usage:
    python upload_to_hub.py <checkpoint_path> <repo_id>
Example:
    python upload_to_hub.py /path/to/checkpoint.th username/model_name
"""

import argparse
import torch
from transformers import AutoProcessor, AutoModel
from audiocraft.models import CompressionModel
import os
from huggingface_hub import HfApi, login, whoami, create_repo
import json
import sys

def check_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        user_info = whoami()
        return user_info["name"]
    except Exception:
        return None

def upload_model(checkpoint_path: str, repo_id: str):
    # Check authentication
    username = check_auth()
    if not username:
        print("❌ Not authenticated with Hugging Face Hub.")
        print("Please run 'huggingface-cli login' first and enter your token.")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        return 1

    # Ensure repo_id includes username
    if '/' not in repo_id:
        repo_id = f"{username}/{repo_id}"
        print(f"Using repository ID: {repo_id}")

    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load base EnCodec model
    print(f"Loading base EnCodec model...")
    model = CompressionModel.get_pretrained('facebook/encodec_24khz')

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "best_state" in checkpoint:
        state_dict = checkpoint["best_state"]
    else:
        raise ValueError("Checkpoint format not recognized. Expected 'model' or 'best_state' key.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("⚠️ Missing keys in the checkpoint that were not initialized in the model:")
        for k in missing_keys:
            print(f"  - {k}")
    if unexpected_keys:
        print("⚠️ Unexpected keys in the checkpoint that do not match the model:")
        for k in unexpected_keys:
            print(f"  - {k}")
    if not missing_keys and not unexpected_keys:
        print("✅ All model weights loaded successfully.")

    # Create local directory if it doesn't exist
    os.makedirs(repo_id, exist_ok=True)

    # Save model in Hugging Face-compatible format
    print("Saving model in Hugging Face format...")
    from audiocraft.utils import export as ac_export
    ac_export.export_encodec(checkpoint_path, os.path.join(repo_id, "compression_state_dict.bin"))

    # Save model config
    print("Saving model config...")
    config = {
        "model_type": "encodec",
        "sample_rate": 24000,
        "channels": 1,
        "hidden_size": 128,
        "num_filters": 32,
        "kernel_size": 7,
        "stride": 2,
        "num_residual_layers": 1,
        "num_embeddings": 1024,
        "embedding_dim": 128,
        "use_conv_shortcut": True
    }
    with open(os.path.join(repo_id, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load and save processor
    print("Loading and saving processor...")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    processor.save_pretrained(repo_id)

    # Create README.md
    print("Creating README.md...")
    readme_content = f"""---
language: en
tags:
- audio
- encodec
- audio-compression
license: mit
---

# {repo_id}

This repository contains weights for a fine-tuned EnCodec model trained using AudioCraft.

## Usage

```python
from audiocraft.models import CompressionModel
from transformers import AutoProcessor

# Load model and processor
model = CompressionModel.get_pretrained("{repo_id}/compression_state_dict.bin")
processor = AutoProcessor.from_pretrained("{repo_id}")
```
"""
    with open(os.path.join(repo_id, "README.md"), "w") as f:
        f.write(readme_content)

    # Create repository if it doesn't exist
    print("Creating repository on Hugging Face Hub...")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating repository: {str(e)}")
        print("Make sure you have the necessary permissions and the repository name is valid.")
        return 1

    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    api = HfApi()
    try:
        api.upload_folder(
            folder_path=repo_id,
            repo_id=repo_id,
            repo_type="model"
        )
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face Hub: {str(e)}")
        print("Make sure the repository exists and you have the necessary permissions.")
        return 1

    print(f"✅ Model and processor successfully uploaded to: https://huggingface.co/{repo_id}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Upload an EnCodec model checkpoint to the Hugging Face Hub")
    parser.add_argument("checkpoint", help="Path to the .th checkpoint file")
    parser.add_argument("repo_id", help="Hugging Face repo name (will be prefixed with username if not provided)")
    args = parser.parse_args()

    try:
        return upload_model(args.checkpoint, args.repo_id)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 