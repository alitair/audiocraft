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
from huggingface_hub import HfApi, login
import json

def upload_model(checkpoint_path: str, repo_id: str):
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

    model.load_state_dict(state_dict, strict=False)

    # Create local directory if it doesn't exist
    os.makedirs(repo_id, exist_ok=True)

    # Save model state dict
    print(f"Saving model state dict...")
    torch.save(model.state_dict(), os.path.join(repo_id, "pytorch_model.bin"))

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

    # Push to Hugging Face Hub
    print("Pushing to Hugging Face Hub...")
    api = HfApi()
    api.upload_folder(
        folder_path=repo_id,
        repo_id=repo_id,
        repo_type="model"
    )

    print(f"✅ Model and processor successfully uploaded to: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload an EnCodec model checkpoint to the Hugging Face Hub")
    parser.add_argument("checkpoint", help="Path to the .th checkpoint file")
    parser.add_argument("repo_id", help="Hugging Face repo name, e.g., username/model_name")
    args = parser.parse_args()

    try:
        upload_model(args.checkpoint, args.repo_id)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 