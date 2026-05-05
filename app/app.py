import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
from src.recommender import load_assets, expand_query_with_claude, search, explain_recommendations_with_claude

st.set_page_config(page_title="Cold-Start Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Cold-Start Movie Recommender")
st.markdown("Find movies using **semantic search** — no account, no history, no ratings needed.")

# Load assets once and cache
@st.cache_resource
def get_assets():
    return load_assets()

df, index, model = get_assets()

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Find Movies", "🆕 New Movie Concept"])

# --- Tab 1: Find Movies ---
with tab1:
    st.subheader("Describe what you want to watch")
    user_query = st.text_input(
        label="Your description",
        placeholder="e.g. a mind-bending sci-fi thriller about identity and memory"
    )
    use_claude = st.toggle("✨ Enhance query with AI", value=True)
    top_k = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    min_rating = st.slider("Minimum rating", min_value=0.0, max_value=9.0, value=6.0, step=0.5)

    if st.button("Find Movies", type="primary") and user_query:
        with st.spinner("Thinking..."):
            if use_claude:
                expanded = expand_query_with_claude(user_query)
                with st.expander("🤖 AI-expanded query"):
                    st.write(expanded)
                results = search(expanded, df, index, model, top_k, min_rating)
            else:
                results = search(user_query, df, index, model, top_k, min_rating)

            explanation = explain_recommendations_with_claude(user_query, results)

        st.subheader("Recommended Movies")
        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                st.markdown(f"### {row['title']}")
                st.caption(f"⭐ {row['vote_average']:.1f} | 🗓 {str(row['release_date'])[:4]} | 🎭 {row['genres']}")
                st.write(row['overview'][:300] + "...")
                st.progress(float(row['similarity_score']), text=f"Match: {row['similarity_score']:.2f}")
                st.divider()

        st.subheader("🤖 Why these movies?")
        st.markdown(explanation)

# --- Tab 2: New Movie Concept ---
with tab2:
    st.subheader("Pitch a new movie concept")
    st.markdown("Describe an unmade movie — the system finds the most similar existing films.")

    concept_title = st.text_input("Working title", placeholder="e.g. The Last Signal")
    concept_plot = st.text_area(
        "Plot concept",
        placeholder="e.g. A deep-space astronaut receives a distress signal from a ship that disappeared 50 years ago...",
        height=120
    )
    concept_genres = st.multiselect(
        "Genres",
        ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
         "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Science Fiction",
         "Thriller", "Western"],
        default=["Science Fiction", "Thriller"]
    )

    if st.button("Find Similar Movies", type="primary") and concept_plot:
        with st.spinner("Analysing concept..."):
            concept_text = (
                f"{concept_title}. "
                f"{concept_plot} "
                f"Genres: {', '.join(concept_genres)}."
            )
            expanded = expand_query_with_claude(concept_text)
            results = search(expanded, df, index, model, top_k=5)
            explanation = explain_recommendations_with_claude(concept_plot, results)

        with st.expander("🤖 AI interpretation of your concept"):
            st.write(expanded)

        st.subheader("Most Similar Existing Movies")
        for _, row in results.iterrows():
            st.markdown(f"**{row['title']}** ({str(row['release_date'])[:4]}) — {row['genres']}")
            st.caption(f"⭐ {row['vote_average']:.1f} | Match: {row['similarity_score']:.2f}")
            st.write(row['overview'][:250] + "...")
            st.divider()

        st.subheader("🤖 Why these comparisons?")
        st.markdown(explanation)