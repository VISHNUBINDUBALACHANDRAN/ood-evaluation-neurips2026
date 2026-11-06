"""
Utility functions for loading/saving model weights.
Prefers SafeTensors format (no code execution risk) with fallback to .pt.
"""

import os
import torch
from safetensors.torch import save_file, load_file


def save_model_weights(state_dict: dict, path: str):
    """
    Save model weights as SafeTensors.

    Args:
        state_dict: dict of {str: torch.Tensor}
        path: full file path ending in .safetensors
    """
    save_file(state_dict, path)


def load_model_weights(directory: str, stem: str = "model", device: str = "cpu") -> dict:
    """
    Load model weights, preferring .safetensors over .pt.

    Args:
        directory: directory containing the weight file
        stem: filename without extension (e.g. "model", "best_model")
        device: target device string

    Returns:
        state_dict: dict of {str: torch.Tensor}
    """
    st_path = os.path.join(directory, f"{stem}.safetensors")
    pt_path = os.path.join(directory, f"{stem}.pt")
    if os.path.exists(st_path):
        return load_file(st_path, device=device)
    elif os.path.exists(pt_path):
        return torch.load(pt_path, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model weights found: tried {st_path} and {pt_path}"
        )


def model_weights_exist(directory: str, stem: str = "model") -> bool:
    """Check if model weights exist in either format."""
    st_path = os.path.join(directory, f"{stem}.safetensors")
    pt_path = os.path.join(directory, f"{stem}.pt")
    return os.path.exists(st_path) or os.path.exists(pt_path)


def save_mask(tensor: torch.Tensor, path: str):
    """Save a single mask tensor as SafeTensors."""
    save_file({"tensor": tensor}, path)


def load_mask(path: str, device: str = "cpu") -> torch.Tensor:
    """Load a single mask tensor from SafeTensors or .pt."""
    if path.endswith(".safetensors") and os.path.exists(path):
        data = load_file(path, device=device)
        return data["tensor"]
    elif path.endswith(".pt") and os.path.exists(path):
        return torch.load(path, map_location=device, weights_only=True)
    # Try safetensors version of a .pt path
    st_path = path.replace(".pt", ".safetensors")
    if os.path.exists(st_path):
        data = load_file(st_path, device=device)
        return data["tensor"]
    elif os.path.exists(path):
        return torch.load(path, map_location=device, weights_only=True)
    raise FileNotFoundError(f"No mask file found: tried {path} and {st_path}")
