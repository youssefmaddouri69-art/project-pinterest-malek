import numpy as np
from  tes import dil


def show_max(s, device, image, model):
    probs = dil(s, device, image, model)  # get probabilities from your function
    idx = np.argmax(probs)  # index of highest probability
    
    # Return a JSON-serializable dict
    return {
        "best_match": s[idx],
        "confidence": float(probs[idx])  # convert numpy float to Python float
    }