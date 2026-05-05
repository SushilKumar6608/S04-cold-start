#06_coldstart_experiment.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

RAW       = r"D:\Egna projekt\S04-cold-start\data\raw\ml-25m"
PROCESSED = r"D:\Egna projekt\S04-cold-start\data\processed"

print("Loading data...")
movies  = pd.read_csv(f"{RAW}/movies.csv")
ratings = pd.read_csv(f"{RAW}/ratings.csv")
links   = pd.read_csv(f"{RAW}/links.csv")
tmdb_df = pd.read_csv(f"{PROCESSED}/movies_clean.csv")

# --- Step 1: Split old vs new movies ---
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
old_ids = set(movies[movies['year'] <= 2015]['movieId'])
new_ids = set(movies[movies['year'] >  2015]['movieId'])
print(f"Old movies (pre-2015): {len(old_ids)}")
print(f"New movies (post-2015): {len(new_ids)}")

# --- Step 2: Build CF on OLD movies only ---
print("\nBuilding collaborative filter on old movies...")
old_ratings = ratings[ratings['movieId'].isin(old_ids)].copy()

# Sample users for speed (keep users with ≥20 ratings)
user_counts = old_ratings['userId'].value_counts()
active_users = user_counts[user_counts >= 20].index
old_ratings = old_ratings[old_ratings['userId'].isin(active_users)]
print(f"Ratings used for CF: {len(old_ratings):,}")
print(f"Active users: {old_ratings['userId'].nunique():,}")

# Build user-item matrix
user_idx  = {u: i for i, u in enumerate(old_ratings['userId'].unique())}
movie_idx = {m: i for i, m in enumerate(old_ratings['movieId'].unique())}
idx_movie = {i: m for m, i in movie_idx.items()}

rows = old_ratings['userId'].map(user_idx)
cols = old_ratings['movieId'].map(movie_idx)
data = old_ratings['rating'].values

R = csr_matrix((data, (rows, cols)),
               shape=(len(user_idx), len(movie_idx)),
               dtype=np.float32)

# SVD (k=50 latent factors)
print("Running SVD (k=50)...")
U, sigma, Vt = svds(R, k=50)
sigma_diag = np.diag(sigma)
item_factors = (sigma_diag @ Vt).T  # shape: (n_movies, 50)
print(f"Item factors shape: {item_factors.shape}")

# --- Step 3: Link MovieLens → TMDB ---
print("\nLinking MovieLens → TMDB...")
links['tmdbId'] = links['tmdbId'].fillna(0).astype(int).astype(str)
tmdb_df['id']   = tmdb_df['id'].astype(str)

merged = links.merge(tmdb_df[['id', 'title', 'text_for_embedding',
                               'genres', 'overview', 'vote_average']],
                     left_on='tmdbId', right_on='id', how='inner')
merged = merged.merge(movies[['movieId', 'year']], on='movieId', how='inner')
print(f"Movies matched between MovieLens and TMDB: {len(merged)}")

new_merged = merged[merged['year'] > 2015].reset_index(drop=True)
old_merged = merged[merged['year'] <= 2015].reset_index(drop=True)
print(f"New movies with TMDB metadata: {len(new_merged)}")
print(f"Old movies with TMDB metadata: {len(old_merged)}")

# --- Step 4: Embedding-based similarity for new movies ---
print("\nLoading embeddings and FAISS index...")
embeddings = np.load(f"{PROCESSED}/embeddings.npy")
index      = faiss.read_index(f"{PROCESSED}/faiss_index.bin")
tmdb_clean = pd.read_csv(f"{PROCESSED}/movies_clean.csv")

# Build a tmdbId → embedding row lookup
tmdb_clean['id'] = tmdb_clean['id'].astype(str)
tmdb_id_to_row   = {row['id']: i for i, row in tmdb_clean.iterrows()}

# --- Step 5: Evaluate CF vs Embeddings on new movies ---
print("\nEvaluating CF vs Embeddings on new movies...")

def get_embedding_neighbours(tmdb_id, top_k=10):
    """Find top-k similar movies by embedding for a given tmdbId."""
    row = tmdb_id_to_row.get(str(tmdb_id))
    if row is None:
        return []
    q = embeddings[row:row+1].copy()
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k + 1)
    # Exclude self (index 0 is always the movie itself)
    neighbour_ids = [tmdb_clean.iloc[i]['id'] for i in indices[0][1:]]
    return neighbour_ids

def cf_can_handle(ml_movie_id):
    """Check if CF has seen this movie (i.e. it's in the training matrix)."""
    return ml_movie_id in movie_idx

results = []
sample = new_merged.sample(min(200, len(new_merged)), random_state=42)

for _, row in sample.iterrows():
    ml_id   = row['movieId']
    tmdb_id = row['id']
    year    = row['year']

    cf_seen      = cf_can_handle(ml_id)
    ratings_count = len(ratings[ratings['movieId'] == ml_id])
    emb_neighbours = get_embedding_neighbours(tmdb_id, top_k=10)

    results.append({
        'title':            row['title_x'] if 'title_x' in row else row['title'],
        'year':             year,
        'ml_ratings_count': ratings_count,
        'cf_can_handle':    cf_seen,
        'emb_neighbours':   len(emb_neighbours),
        'vote_average':     row['vote_average'],
    })

results_df = pd.DataFrame(results)

print("\n=== Cold-Start Evaluation Summary ===")
print(f"New movies sampled:              {len(results_df)}")
print(f"CF can handle (in train matrix): {results_df['cf_can_handle'].sum()} "
      f"({results_df['cf_can_handle'].mean()*100:.1f}%)")
print(f"CF cannot handle (cold-start):   {(~results_df['cf_can_handle']).sum()} "
      f"({(~results_df['cf_can_handle']).mean()*100:.1f}%)")
print(f"Embedding always finds neighbours: "
      f"{(results_df['emb_neighbours'] > 0).sum()} "
      f"({(results_df['emb_neighbours'] > 0).mean()*100:.1f}%)")
print(f"\nAvg ratings for new movies CF can handle:    "
      f"{results_df[results_df['cf_can_handle']]['ml_ratings_count'].mean():.1f}")
print(f"Avg ratings for new movies CF cannot handle: "
      f"{results_df[~results_df['cf_can_handle']]['ml_ratings_count'].mean():.1f}")

results_df.to_csv(f"{PROCESSED}/coldstart_evaluation.csv", index=False)
print(f"\nSaved to data/processed/coldstart_evaluation.csv")

# Show some examples
print("\n=== Example new movies CF cannot handle (embedding fills the gap) ===")
cold = results_df[~results_df['cf_can_handle']].head(8)
print(cold[['title', 'year', 'ml_ratings_count', 'vote_average']].to_string(index=False))