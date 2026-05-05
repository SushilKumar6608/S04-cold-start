# 🎬 S04 — Content Cold-Start Solver using Metadata Embeddings

Embedding-based recommender system that solves the **item cold-start problem** — recommending new content before any user interaction data exists. Benchmarked against a collaborative filtering baseline (SVD) on MovieLens 25M to quantify exactly where CF fails and where the embedding system succeeds.

> **Real-world relevance:** Every new listing on a platform like Sellpy is a unique second-hand item with zero interaction history. Collaborative filtering cannot recommend it. This system can — immediately, using only the item's text description and attributes.

---

## The Cold-Start Problem

Traditional recommenders fail for new content:

| Method | How it works | Fails when |
|--------|-------------|------------|
| Collaborative Filtering (SVD) | Learns latent factors from user-item interactions | Item has zero or too few ratings |
| Popularity baseline | Returns most-rated items | Always ignores new content |
| **This system** | Embeds item metadata into semantic vector space | Never — works from day one |

### Quantified on MovieLens 25M

| Metric | Value |
|--------|-------|
| New movies (post-2015) in MovieLens | 7,890 |
| CF coverage on new movies | **0%** |
| Embedding system coverage on new movies | **100%** |
| New movies with fewer than 50 ratings | **88.2%** |
| FAISS search latency | **7ms** |
| Movies indexed | 44,937 |

---

## Pipeline

```
TMDB Metadata (1.4M movies)
        ↓
  Quality Filter (English · vote_count ≥ 10 · overview present)
        ↓
  44,937 movies retained
        ↓
  Text Construction
  title + tagline + overview + "Genres: " + genres + "Keywords: " + keywords
        ↓
  all-MiniLM-L6-v2 Sentence Embeddings  →  (44937, 384) float32
        ↓
  FAISS IndexFlatIP (L2-normalised cosine similarity)  →  7ms search
        ↓
  Claude Sonnet (query expansion + recommendation explanation)
        ↓
  Streamlit App (3-tab interface)
```

---

## Cold-Start Experiment

**Baseline:** SVD collaborative filter trained on MovieLens 25M ratings for pre-2015 movies only.

**Test set:** 7,890 post-2015 movies — the cold-start zone.

**Results:**
- CF trained on 24.4M ratings from 161,557 users
- Post-2015 movies were never seen during training → **0% coverage**
- Embedding system finds neighbours for **100% of new movies** using only metadata
- 88.2% of new movies have fewer than 50 ratings — even if CF saw them, it would have almost no signal

**Example — "I'm Not Ashamed" (2016, only 7 ratings):**

CF: *Cannot recommend — item not in training matrix.*

Embedding system returns:
1. We Are Columbine — similarity 0.60
2. She Fought Alone — similarity 0.57
3. Bad Reputation — similarity 0.55
4. The Hunting Ground — similarity 0.54
5. The Unfaithful — similarity 0.53

All thematically aligned. Zero ratings needed.

---

## App Features

**Tab 1 — Cold-Start Demo**
- Pick any new movie (post-2015) from a dropdown
- Side-by-side: CF failure panel vs Embedding success panel
- Live similarity search with match scores
- Evidence plots embedded in the app

**Tab 2 — Find Movies**
- Natural language query → top-K recommendations
- Toggle AI query enhancement (Claude Sonnet)
- Minimum rating filter + recommendation count slider
- Claude explanation of why each movie was recommended

**Tab 3 — New Movie Concept**
- Pitch an unmade movie: title + plot + genre selection
- System finds most similar existing movies using metadata only
- Pure cold-start: zero ratings, pure description

---

## Project Structure

```
S04-cold-start/
├── data/
│   ├── raw/                         # TMDB CSV + MovieLens 25M (not committed)
│   └── processed/
│       ├── movies_clean.csv         # 44,937 filtered movies
│       ├── coldstart_evaluation.csv # CF vs Embedding experiment results
│       ├── embeddings.npy           # (44937, 384) float32 (not committed)
│       └── faiss_index.bin          # FAISS index (not committed)
├── notebooks/
│   ├── 01_exploration.py            # TMDB dataset overview
│   ├── 02_cleaning.py               # Filter and construct embedding text
│   ├── 03_embeddings.py             # Generate MiniLM embeddings (~9 min CPU)
│   ├── 04_build_index.py            # Build FAISS index
│   ├── 05_movielens_explore.py      # MovieLens 25M exploration
│   ├── 06_coldstart_experiment.py   # CF vs Embedding cold-start evaluation
│   └── 08_coldstart_plots_white.py  # White-theme plots (used in docs)
├── src/
│   └── recommender.py               # Core recommender logic
├── app/
│   └── app.py                       # Streamlit 3-tab application
├── docs/
│   ├── plot_coldstart_coverage.png  # CF vs Embedding coverage chart
│   ├── plot_ratings_scarcity.png    # Rating scarcity distribution
│   ├── plot_embedding_example.png   # Cold-start example with neighbours
│   ├── s04_dashboard.html           # Project dashboard
│   └── s04_linkedin_thumbnail.html  # LinkedIn thumbnail
├── .env                             # ANTHROPIC_API_KEY (not committed)
├── .gitignore
└── README.md
```

---

## Setup

**1. Clone and create environment**
```bash
git clone https://github.com/SushilKumar6608/S04-cold-start.git
cd S04-cold-start
conda create --prefix ./env python=3.11 -y
conda activate ./env
pip install numpy pandas scikit-learn sentence-transformers faiss-cpu streamlit anthropic python-dotenv scipy ipykernel
```

**2. Download datasets**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
cd data/raw
kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies --unzip
kaggle datasets download -d garymk/movielens-25m-dataset --unzip
```

**3. Build the pipeline**
```bash
python notebooks/02_cleaning.py
python notebooks/03_embeddings.py            # ~9 min on CPU
python notebooks/04_build_index.py
python notebooks/06_coldstart_experiment.py  # CF baseline + evaluation
```

**4. Set API key**
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

**5. Run the app**
```bash
streamlit run app/app.py
```

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Datasets | TMDB (1.4M movies) · MovieLens 25M (25M ratings · 162k users) |
| CF Baseline | scipy SVD (k=50 latent factors) · trained on pre-2015 ratings |
| Embeddings | sentence-transformers · all-MiniLM-L6-v2 · 384-dim |
| Vector search | FAISS · IndexFlatIP · cosine similarity · 7ms latency |
| LLM | Anthropic Claude Sonnet · query expansion + explanations |
| App | Streamlit · 3-tab interface |
| Runtime | Python 3.11 · conda |

---

## Real-World Applicability

The pipeline is domain-agnostic — any item catalogue with text metadata can replace TMDB:

- **Sellpy** — every second-hand listing is unique, zero interaction history by definition
- **Viaplay** — new Nordic originals need recommendations before any viewing data exists
- **E-commerce** — new product launches need day-one visibility
- **Job platforms** — new job postings need to surface to relevant candidates immediately

---

## Author

**Ganisetty Sai Surya Sushil Kumar**  
MSc AI & Automation · University West, Sweden  
[github.com/SushilKumar6608](https://github.com/SushilKumar6608)