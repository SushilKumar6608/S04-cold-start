import numpy as np
import faiss

# Load embeddings
embeddings = np.load(r"D:\Egna projekt\S04-cold-start\data\processed\embeddings.npy")
print(f"Embeddings shape: {embeddings.shape}")

# Normalise vectors (needed for cosine similarity with FAISS)
faiss.normalize_L2(embeddings)

# Build the index
dim = embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity after normalisation
index.add(embeddings)
print(f"Index built with {index.ntotal} vectors")

# Save the index
faiss.write_index(index, r"D:\Egna projekt\S04-cold-start\data\processed\faiss_index.bin")
print("Saved to data/processed/faiss_index.bin")