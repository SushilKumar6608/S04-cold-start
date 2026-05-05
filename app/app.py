import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from src.recommender import expand_query_with_claude, explain_recommendations_with_claude

load_dotenv()

st.set_page_config(
    page_title="S04 — Cold-Start Solver",
    page_icon="🎬",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 S04 Cold-Start Solver")
    st.markdown("---")
    st.markdown("""
**The Problem**

Traditional recommenders (collaborative filtering) need user interaction history.
New items with zero ratings get ignored completely.

**The Solution**

Semantic embeddings from item metadata.
New content can be recommended immediately — no ratings required.
    """)
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- TMDB · 44,937 movies")
    st.markdown("- MovieLens 25M · 162k users")
    st.markdown("**Models**")
    st.markdown("- CF baseline: SVD (k=50)")
    st.markdown("- Ours: all-MiniLM-L6-v2 + FAISS")
    st.markdown("---")
    st.markdown("**Key result**")
    st.metric("CF coverage on new movies", "0%")
    st.metric("Embedding coverage on new movies", "100%")

# ── Load assets ───────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    PROCESSED = r"D:\Egna projekt\S04-cold-start\data\processed"
    df        = pd.read_csv(f"{PROCESSED}/movies_clean.csv")
    index     = faiss.read_index(f"{PROCESSED}/faiss_index.bin")
    model     = SentenceTransformer('all-MiniLM-L6-v2')
    emb       = np.load(f"{PROCESSED}/embeddings.npy")
    eval_df   = pd.read_csv(f"{PROCESSED}/coldstart_evaluation.csv")
    df['id']  = df['id'].astype(str)
    id_to_row = {row['id']: i for i, row in df.iterrows()}
    return df, index, model, emb, eval_df, id_to_row

df, index, model, embeddings, eval_df, id_to_row = load_assets()

def search(query_text, top_k=5, min_rating=6.0):
    q = model.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, top_k * 10)
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    results = results[results['vote_average'] >= min_rating]
    results = results[results['vote_count'] >= 50]
    return results.head(top_k)[['title', 'genres', 'overview',
                                 'vote_average', 'vote_count',
                                 'release_date', 'similarity_score']]

def render_movie_cards(results):
    cols = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 2]:
            st.markdown(f"### {row['title']}")
            st.caption(
                f"⭐ {row['vote_average']:.1f} · "
                f"🗓 {str(row['release_date'])[:4]} · "
                f"🎭 {row['genres']}"
            )
            st.write(row['overview'][:280] + "...")
            st.progress(
                float(np.clip(row['similarity_score'], 0, 1)),
                text=f"Match score: {row['similarity_score']:.3f}"
            )
            st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧊 Cold-Start Demo",
    "🔍 Find Movies",
    "🆕 New Movie Concept"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Cold-Start Demo
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Cold-Start Problem: CF vs Embedding System")
    st.markdown(
        "Pick any **new movie** (post-2015). "
        "See how collaborative filtering fails — and how our embedding system fills the gap."
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("New movies (post-2015)", "7,890")
    col2.metric("CF coverage on new movies", "0%", delta="-100%", delta_color="inverse")
    col3.metric("Embedding coverage", "100%", delta="+100%")
    col4.metric("Avg ratings (new movies)", "67", help="vs 452 for pre-2015 movies")

    st.divider()

    # Movie selector
    cold_movies = eval_df.sort_values('ml_ratings_count').reset_index(drop=True)
    cold_movies['display'] = cold_movies.apply(
        lambda r: f"{r['title']} ({int(r['year'])} · {int(r['ml_ratings_count'])} ratings)", axis=1
    )

    selected = st.selectbox(
        "Choose a cold-start movie:",
        cold_movies['display'].tolist(),
        index=0
    )

    selected_row = cold_movies[cold_movies['display'] == selected].iloc[0]

    col_a, col_b = st.columns(2)

    # CF panel
    with col_a:
        st.markdown("### ❌ Collaborative Filtering (SVD)")
        st.error(
            f"**Cannot recommend.**\n\n"
            f"'{selected_row['title']}' has only "
            f"**{int(selected_row['ml_ratings_count'])} ratings** in MovieLens. "
            f"It was never seen during training — CF has no latent factors for this item. "
            f"Any user asking for movies similar to this gets nothing."
        )
        st.markdown("**Why CF fails here:**")
        st.markdown("""
- SVD requires the item to exist in the training interaction matrix
- New items with few ratings have unreliable or missing latent vectors  
- 87% of post-2015 movies have fewer than 50 ratings — all invisible to CF
        """)

    # Embedding panel
    with col_b:
        st.markdown("### ✅ Embedding System (ours)")

        match = df[df['title'].str.contains(
            selected_row['title'].split(':')[0].strip(),
            case=False, na=False
        )]

        if len(match) > 0:
            row_idx = id_to_row.get(str(match.iloc[0]['id']))
            if row_idx is not None:
                q = embeddings[row_idx:row_idx+1].copy()
                faiss.normalize_L2(q)
                scores, indices = index.search(q, 6)
                neighbours = []
                for i, s in zip(indices[0][1:], scores[0][1:]):
                    r = df.iloc[i]
                    neighbours.append({
                        'title': r['title'],
                        'genres': r['genres'],
                        'score': float(s),
                        'vote_average': r['vote_average'],
                        'release_date': str(r['release_date'])[:4]
                    })

                st.success(f"Found **{len(neighbours)} similar movies** using metadata only.")
                for n in neighbours:
                    st.markdown(
                        f"**{n['title']}** ({n['release_date']}) · "
                        f"⭐ {n['vote_average']:.1f} · {n['genres'][:40]}"
                    )
                    st.progress(
                        float(np.clip(n['score'] * 1.2, 0, 1)),
                        text=f"Similarity: {n['score']:.3f}"
                    )
            else:
                st.info("Movie not found in TMDB metadata. Try another.")
        else:
            st.info("Movie not found in TMDB metadata. Try another.")

    # Plots
    st.divider()
    st.subheader("📊 Evidence")
    pc1, pc2 = st.columns(2)
    with pc1:
        st.image(r"D:\Egna projekt\S04-cold-start\docs\plot_coldstart_coverage.png")
    with pc2:
        st.image(r"D:\Egna projekt\S04-cold-start\docs\plot_ratings_scarcity.png")
    st.image(r"D:\Egna projekt\S04-cold-start\docs\plot_embedding_example.png")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Find Movies
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Find Movies by Description")
    st.markdown("Describe what you want to watch in natural language.")

    query      = st.text_input("Your description",
                               placeholder="e.g. a slow-burn psychological thriller set in Scandinavia")
    use_claude = st.toggle("✨ Enhance query with AI", value=True)
    col1, col2 = st.columns(2)
    top_k      = col1.slider("Recommendations", 3, 10, 5)
    min_rating = col2.slider("Min rating", 0.0, 9.0, 6.0, 0.5)

    if st.button("Find Movies", type="primary") and query:
        with st.spinner("Searching..."):
            if use_claude:
                expanded = expand_query_with_claude(query)
                with st.expander("🤖 AI-expanded query"):
                    st.write(expanded)
                results = search(expanded, top_k, min_rating)
            else:
                results = search(query, top_k, min_rating)

            explanation = explain_recommendations_with_claude(query, results)

        render_movie_cards(results)
        st.subheader("🤖 Why these movies?")
        st.markdown(explanation)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — New Movie Concept
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Pitch a New Movie Concept")
    st.markdown(
        "Describe an **unmade movie**. "
        "The system finds the most similar existing films using metadata only — "
        "zero ratings required. This is cold-start in action."
    )

    concept_title = st.text_input("Working title",
                                   placeholder="e.g. The Last Diagnosis")
    concept_plot  = st.text_area("Plot concept", height=100,
                                  placeholder="e.g. A burnt-out surgeon discovers a pattern of unexplained deaths...")
    concept_genres = st.multiselect("Genres",
        ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
         "Drama", "Fantasy", "Horror", "Mystery", "Romance",
         "Science Fiction", "Thriller", "Western"],
        default=["Drama", "Thriller"]
    )

    if st.button("Find Similar Movies", type="primary") and concept_plot:
        with st.spinner("Analysing concept..."):
            concept_text = (
                f"{concept_title}. {concept_plot} "
                f"Genres: {', '.join(concept_genres)}."
            )
            expanded = expand_query_with_claude(concept_text)
            results  = search(expanded, top_k=5, min_rating=6.0)
            explanation = explain_recommendations_with_claude(concept_plot, results)

        with st.expander("🤖 AI interpretation of your concept"):
            st.write(expanded)

        render_movie_cards(results)
        st.subheader("🤖 Why these comparisons?")
        st.markdown(explanation)