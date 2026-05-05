import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import anthropic
import os

load_dotenv()

# Load all assets once
def load_assets():
    df = pd.read_csv(r"D:\Egna projekt\S04-cold-start\data\processed\movies_clean.csv")
    index = faiss.read_index(r"D:\Egna projekt\S04-cold-start\data\processed\faiss_index.bin")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, index, model

def expand_query_with_claude(user_query: str) -> str:
    """Use Claude to expand a vague user query into rich descriptive text for better embedding."""
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    prompt = f"""You are helping a movie recommendation system. 
A user described what they want to watch as: "{user_query}"

Expand this into a rich, detailed movie description (3-4 sentences) that includes:
- Likely plot themes and narrative style
- Probable genres
- Mood and tone
- Likely keywords

Return only the expanded description, no preamble."""

    msg = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=200,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return msg.content[0].text.strip()


def search(query_text: str, df, index, model, top_k: int = 5, min_rating: float = 6.0, min_votes: int = 50) -> pd.DataFrame:
    """Embed a query and find top-k similar movies using FAISS."""
    # Search more candidates than needed, then filter by quality
    query_vec = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k * 10)  # fetch 50, filter to 5

    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]

    # Filter by minimum quality
    results = results[results['vote_average'] >= min_rating]
    results = results[results['vote_count'] >= min_votes]

    return results.head(top_k)[['title', 'genres', 'overview', 'vote_average',
                                 'vote_count', 'release_date', 'similarity_score']]


def explain_recommendations_with_claude(user_query: str, recommendations: pd.DataFrame) -> str:
    """Use Claude to explain why each movie was recommended."""
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    movies_text = ""
    for i, row in recommendations.iterrows():
        movies_text += f"{row['title']} ({row['genres']}) — {row['overview'][:150]}...\n"

    prompt = f"""A user wanted: "{user_query}"

These movies were recommended by a semantic similarity search:
{movies_text}

For each movie, write one sentence explaining why it matches what the user wanted.
Format: "**Movie Title** — reason"."""

    msg = client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=400,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return msg.content[0].text.strip()


if __name__ == "__main__":
    print("Loading assets...")
    df, index, model = load_assets()

    # Test 1: Direct movie title
    print("\n--- Test 1: Search by movie title ---")
    results = search("A mind-bending thriller about dreams within dreams", df, index, model)
    print(results[['title', 'genres', 'similarity_score']].to_string(index=False))

    # Test 2: Vague query expanded by Claude
    print("\n--- Test 2: Vague query expanded by Claude ---")
    user_query = "something scary but also makes me think"
    print(f"User query: '{user_query}'")
    expanded = expand_query_with_claude(user_query)
    print(f"Expanded: {expanded}")
    results2 = search(expanded, df, index, model)
    print(results2[['title', 'genres', 'similarity_score']].to_string(index=False))

    # Test 3: Explanation
    print("\n--- Test 3: Claude explains recommendations ---")
    explanation = explain_recommendations_with_claude(user_query, results2)
    print(explanation)