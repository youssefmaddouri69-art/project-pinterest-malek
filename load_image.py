

# load_image.py
from PIL import Image
import torch

def load_image(image_path: str, preprocess, device: str) -> torch.Tensor:
    """
    Load an image from disk, apply CLIP preprocessing, and move it to the device.

    Args:
        image_path: Path to the image file
        preprocess: CLIP preprocessing function
        device: 'cuda' or 'cpu'

    Returns:
        Preprocessed image tensor ready for CLIP
    """
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure 3 channels
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")
