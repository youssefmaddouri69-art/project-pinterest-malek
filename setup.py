import torch, clip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Set up CLIP model and preprocessing pipeline

def setup(model_name="ViT-B/32"):
    # Set up device, model, and preprocessing
    #what  this code does is initialize the CLIP model and its preprocessing pipeline.
    """
    Set up CLIP model and preprocessing pipeline.
    
    Args:
        model_name (str): CLIP model variant (default: "ViT-B/32")
        
    Returns:
        tuple: (device, preprocess, model, embedding_dim)
    """
    # Pick device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{model_name}' on {device}...")

    try:
        # Load CLIP once
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        # Embedding dimension of the visual encoder
        embedding_dim = model.visual.output_dim  

        logger.info(f"Model {model_name} loaded successfully on {device}. "
                    f"Embedding dimension: {embedding_dim}")
        
        return device, preprocess, model, embedding_dim

    except Exception as e:
        logger.error(f"Failed to load CLIP model {model_name}: {e}")
        raise


