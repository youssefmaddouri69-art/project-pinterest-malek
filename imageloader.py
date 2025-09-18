import torch
from PIL import Image
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
# Function to load and preprocess an image


# Function to load and preprocess an image
def load_image(path, preprocess, device) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Always return tensor
            return preprocess(image).to(device)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        # Return a placeholder tensor so DataLoader can continue
        return torch.zeros(1, 3, 224, 224, device=device)
        
# Dataset class for loading images
class ImageDataset(Dataset):
    def __init__(self, image_paths: List[Path], preprocess, device):
        self.image_paths = image_paths
        self.preprocess = preprocess
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        tensor = load_image(self.image_paths[idx], self.preprocess, self.device)
        return tensor, str(self.image_paths[idx]), tensor is not None
    



from pathlib import Path
from typing import List

def getimagepaths(folder_path: str, extensions: List[str] = None) -> List[str]:
    """
    Get a list of image file paths from a folder.

    Args:
        folder_path: Path to the folder
        extensions: List of file extensions to include (default common image types)

    Returns:
        List of image paths as strings
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']

    folder = Path(folder_path)
    image_paths = [str(p) for ext in extensions for p in folder.rglob(f'*{ext}')]
    
    return image_paths

import os
from typing import List

def get_folders(path: str) -> List[str]:
    """
    Get all folder paths in the given directory.

    Args:
        path: Path to the root directory

    Returns:
        List of full folder paths
    """
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.path)
    return folders
