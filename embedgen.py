import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import clip
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import os
import numpy as np
import json
from pathlib import Path

# Import your existing modules
from imageloader import ImageDataset, getimagepaths, get_folders
from setup import setup
from normalerr import collate_fn
from faissy import faiss_res
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FashionSearchEngine:
    """
    A fashion search engine using CLIP embeddings and FAISS indexing.
    
    This class provides functionality to:
    - Generate embeddings for fashion images
    - Build searchable indices
    - Perform text and image-based searches
    - Save and load processed data
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize the Fashion Search Engine.
        
        Args:
            model_name: CLIP model name to use for embeddings
        """
        self.model_name = model_name
        self.device = None
        self.preprocess = None
        self.model = None
        self.embedding_dim = None
        self.index = None
        self.metadata = None
        
        self._setup_model()
    
    def _setup_model(self):
        """Initialize CLIP model and preprocessing."""
        self.device, self.preprocess, self.model, self.embedding_dim = setup(self.model_name)
        logger.info(f"Model initialized: {self.model_name} on {self.device}")
    
    def generate_embeddings(self, image_paths: List[str], 
                          batch_size: int = 32, 
                          num_workers: int = 0) -> torch.Tensor:
        """
        Generate CLIP embeddings for a list of image paths.
        
        Args:
            image_paths: List of paths to image files
            batch_size: Batch size for processing
            num_workers: Number of worker threads for data loading
            
        Returns:
            Tensor containing normalized embeddings
        """
        dataset = ImageDataset(image_paths, self.preprocess, self.device)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
            collate_fn=collate_fn
        )

        all_embeddings = []
        processed_count = 0
        failed_count = 0

        for batch_images in tqdm(dataloader, desc="Generating embeddings"):
            if len(batch_images) == 0:
                failed_count += batch_size
                continue
                
            try:
                batch_images = batch_images.to(self.device)
                with torch.no_grad():
                    if self.device == "cuda":
                        with torch.amp.autocast("cuda"):
                            batch_embeddings = self.model.encode_image(batch_images)
                    else:
                        batch_embeddings = self.model.encode_image(batch_images)

                # Normalize embeddings
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(batch_embeddings.cpu())
                processed_count += batch_embeddings.size(0)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                failed_count += batch_images.size(0)
                continue

        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            logger.warning("No valid embeddings were generated.")
            return torch.empty((0, self.embedding_dim))

        logger.info(f"Finished generating embeddings. Processed: {processed_count}, Failed: {failed_count}")
        logger.info(f"Final embeddings shape: {final_embeddings.shape}")

        if final_embeddings.shape[0] != len(image_paths):
            logger.warning(f"Size mismatch: Expected {len(image_paths)}, got {final_embeddings.shape[0]}")

        return final_embeddings
    
    def process_folders(self, root_folder: str, max_folders: Optional[int] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Process a directory of fashion image folders to generate embeddings and metadata.
        
        Args:
            root_folder: Path to root directory containing image folders
            max_folders: Maximum number of folders to process (None for all)
            
        Returns:
            Tuple of (embeddings tensor, metadata list)
        """
        folder_list = get_folders(root_folder)
        
        if max_folders is not None:
            folder_list = folder_list[:max_folders]
        
        all_embeddings = []
        all_metadata = []

        for i, folder in enumerate(folder_list):
            logger.info(f"Processing folder {i+1}/{len(folder_list)}: {folder}")

            folder_name = os.path.basename(folder)
            keywords = folder_name.split("_")
            
            image_paths = getimagepaths(folder)
            if not image_paths:
                logger.warning(f"No images found in folder: {folder}")
                continue
                
            embeddings = self.generate_embeddings(image_paths, batch_size=2, num_workers=0)

            if embeddings.shape[0] > 0:
                all_embeddings.append(embeddings)
                
                # Create metadata for successfully processed images
                for j, img_path in enumerate(image_paths[:embeddings.shape[0]]):
                    all_metadata.append({
                        "image_path": img_path,
                        "keywords": keywords,
                        "folder": folder_name,
                        "index": len(all_metadata)
                    })

        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            logger.info(f"Final embeddings shape: {final_embeddings.shape}")
            return final_embeddings, all_metadata
        else:
            logger.error("No embeddings generated from any folder!")
            return torch.empty((0, self.embedding_dim)), []
    
    def build_index(self, embeddings: torch.Tensor, metadata: List[Dict]):
        """
        Build FAISS index from embeddings and store metadata.
        
        Args:
            embeddings: Tensor of embeddings
            metadata: List of metadata dictionaries
        """
        self.index = faiss_res(embeddings, metadata)
        self.metadata = metadata
        logger.info(f"Built FAISS index with {len(metadata)} items")
    
    def encode_text_query(self, text: str) -> np.ndarray:
        """
        Encode text query into embedding space.
        
        Args:
            text: Text query string
            
        Returns:
            Normalized embedding array
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype("float32")
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode single image into embedding space.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding array
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype("float32")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar items using query embedding.
        
        Args:
            query_embedding: Query embedding array
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    
    def search_by_text(self, text: str, k: int = 5) -> List[Dict]:
        """
        Search for similar items using text query.
        
        Args:
            text: Text query
            k: Number of results to return
            
        Returns:
            List of result dictionaries with image paths and metadata
        """
        query_embedding = self.encode_text_query(text)
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distances[0][i])
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def search_by_image(self, image_path: str, k: int = 5) -> List[Dict]:
        """
        Search for similar items using image query.
        
        Args:
            image_path: Path to query image
            k: Number of results to return
            
        Returns:
            List of result dictionaries with image paths and metadata
        """
        query_embedding = self.encode_image(image_path)
        distances, indices = self.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['distance'] = float(distances[0][i])
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save_data(self, embeddings: torch.Tensor, metadata: List[Dict], output_dir: str = "./saved_data"):
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings: Embeddings tensor to save
            metadata: Metadata list to save
            output_dir: Directory to save files in
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save embeddings as numpy array
        embeddings_np = embeddings.float().numpy().astype(np.float32)
        np.save(output_path / "embeddings.npy", embeddings_np)

        # Save metadata as JSON
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved embeddings: {embeddings_np.shape}")
        logger.info(f"Saved metadata: {len(metadata)} entries")
        logger.info(f"Files saved to: {output_path}")
    
    def load_data(self, data_dir: str = "./saved_data"):
        """
        Load previously saved embeddings and metadata.
        
        Args:
            data_dir: Directory containing saved files
        """
        data_path = Path(data_dir)
        
        # Load embeddings
        embeddings_path = data_path / "embeddings.npy"
        if embeddings_path.exists():
            embeddings_np = np.load(embeddings_path)
            embeddings = torch.from_numpy(embeddings_np)
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        # Load metadata
        metadata_path = data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Build index
        self.build_index(embeddings, metadata)
        logger.info(f"Loaded data: {embeddings.shape[0]} items")


def main():
    """Example usage of the Fashion Search Engine."""
    
    # Initialize the search engine
    engine = FashionSearchEngine(model_name="ViT-B/32")
    
    # Configuration
    root_folder = "img"  # Change this to your image folder path
    max_folders = 10     # Process first 10 folders, or None for all
    
    # Process folders and generate embeddings
    logger.info("Processing image folders...")
    embeddings, metadata = engine.process_folders(root_folder, max_folders)
    
    if embeddings.shape[0] == 0:
        logger.error("No embeddings generated. Check your image folder path.")
        return
    
    # Build search index
    logger.info("Building search index...")
    engine.build_index(embeddings, metadata)
    
    # Save processed data
    logger.info("Saving data...")
    engine.save_data(embeddings, metadata)
    
    # Example searches
    logger.info("\nPerforming example searches...")
    
    # Text search
    text_results = engine.search_by_text("dress", k=5)
    logger.info(f"\nText search results for 'dress':")
    for result in text_results:
        logger.info(f"  Rank {result['rank']}: {result['image_path']} (distance: {result['distance']:.3f})")
    
    # Image search (if you have a query image)
    # image_results = engine.search_by_image("path/to/query/image.jpg", k=5)
    
    logger.info("\nSearch engine ready!")
    return engine


if __name__ == "__main__":
    search_engine = main()
# Simple usage
engine = FashionSearchEngine()
embeddings, metadata = engine.process_folders("img", max_folders=5)
engine.build_index(embeddings, metadata)
results = engine.search_by_text("red dress", k=5)
