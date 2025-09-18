import faiss
import numpy as np
import os



def buildfaiss(embedding, index_path="faiss_index.index"):


    # Ensure input is NumPy float32
    if isinstance(embedding, np.ndarray):
        embeddings_np = embedding.astype(np.float32)
    else:  # torch.Tensor
        embeddings_np = embedding.detach().cpu().numpy().astype(np.float32)

    d = embeddings_np.shape[1]

    # Build index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_np)

    print("Total number of embeddings:", index.ntotal)
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")

    return index




def faiss_res(final_embeddings, all_metadata):
    if final_embeddings is None or final_embeddings.shape[0] == 0:
        print("No embeddings to build FAISS index.")
        return None

    index = buildfaiss(final_embeddings.numpy().astype(np.float32))
    print("FAISS index built.")

    return index
