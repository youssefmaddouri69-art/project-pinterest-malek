import torch, clip
import numpy as np
from PIL import Image


#s is list of prompts
#returns list of probabilities
def dil(s,device,image,model):
    prompts = s
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    return probs