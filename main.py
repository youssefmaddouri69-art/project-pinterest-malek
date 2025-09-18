import torch, clip
import numpy as np
from PIL import Image
from load_image import load_image
from tes import dil
from probs import show_max
from categories import clothing, colors  # Import both lists

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model, preprocess = clip.load("ViT-B/32", device=device)
print("Model loaded successfully!")

def main1(path):
    try:
        image = load_image(path, preprocess, device)
        
        # Get clothing type
        clothing_result = show_max(clothing, device, image, model)
        
        # Get color
        color_result = show_max(colors, device, image, model)
        
        # Ensure JSON-serializable
        result_serialized = {
            "clothing": {
                "best_match": str(clothing_result["best_match"]),
                "confidence": float(clothing_result["confidence"])
            },
            "color": {
                "best_match": str(color_result["best_match"]),
                "confidence": float(color_result["confidence"])
            }
        }
        
        return {"result": result_serialized}
    
    except Exception as e:
        raise RuntimeError(f"Error processing image {path}: {str(e)}")

def analyze_categories(path, category_list):
    """Generic function to analyze any category list"""
    try:
        image = load_image(path, preprocess, device)
        result = show_max(category_list, device, image, model)
        return result
    except Exception as e:
        raise RuntimeError(f"Error analyzing categories for {path}: {str(e)}")

if __name__ == "__main__":
    import os
    test_image = "your_image.jpg"
    if os.path.exists(test_image):
        result = main1(test_image)
        print(result)
    else:
        print(f"Test image {test_image} not found")