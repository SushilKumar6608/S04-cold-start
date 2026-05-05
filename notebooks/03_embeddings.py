#03_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# Load cleaned data
df = pd.read_csv(r"D:\Egna projekt\S04-cold-start\data\processed\movies_clean.csv")
print(f"Loaded {len(df)} movies")

# Load the embedding model
print("\nLoading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
print("Generating embeddings (this will take a few minutes)...")
texts = df['text_for_embedding'].tolist()

start = time.time()
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True
)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.1f}s")
print(f"Embeddings shape: {embeddings.shape}")

# Save embeddings
np.save(r"D:\Egna projekt\S04-cold-start\data\processed\embeddings.npy", embeddings)
print("Saved to data/processed/embeddings.npy")