# 🎬 S04 — Content Cold-Start Solver using Metadata Embeddings

Embedding-based recommender system that solves the **cold-start problem** — recommending new content before any user interaction data exists. Uses `sentence-transformers` to embed rich movie metadata into a FAISS vector index, with Claude Sonnet enhancing queries and explaining recommendations.

---

## The Problem

Traditional recommender systems fail for new content:

- **Collaborative filtering** needs user ratings → zero ratings for new items = no recommendations
- **Matrix factorisation** needs interaction history → new items have none
- **Popularity baseline** just returns the same blockbusters forever

This system uses only **item metadata** (plot, genres, keywords, tagline) to build a semantic embedding space. A new movie added today with zero ratings can be recommended immediately.

---

## Pipeline

```
TMDB Metadata (1.4M movies)
        ↓
  Quality Filter (English · vote_count ≥ 10 · overview present)
        ↓
  44,937 movies retained
        ↓
  Text Construction (title + tagline + overview + genres + keywords)
        ↓
  all-MiniLM-L6-v2 Embeddings  →  (44937, 384) float32
        ↓
  FAISS IndexFlatIP (cosine similarity, L2-normalised)
        ↓
  Claude Sonnet (query expansion + recommendation explanation)
        ↓
  Streamlit App (2-tab interface)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Movies indexed | 44,937 |
| Embedding dimensions | 384 |
| Search latency | ~50ms (CPU) |
| Query expansion improvement | +13% similarity score (0.63 → 0.71) |
| Cold-start mode | Zero ratings required |

**Query expansion example:**

> Input: *"something scary but makes me think"*
>
> Claude expands to: *"A psychologically intense horror-thriller that delves into existential dread, moral ambiguity, blending supernatural elements with philosophical questions..."*
>
> Results improve from generic horror (~0.63) to specific psychological thrillers (~0.71)

**New Movie Concept example:**

> Pitched: *"The Last Diagnosis — a burnt-out surgeon in Chicago discovers a pattern of unexplained hospital deaths"*
>
> Found: Coma (1978), The Hospital (1971), The Killing of a Sacred Deer — all thematically aligned, zero ratings needed

---

## App Features

**Tab 1 — Find Movies**
- Natural language query input
- Toggle AI query enhancement (Claude Sonnet)
- Adjustable number of recommendations (3–10)
- Minimum rating filter (default 6.0)
- Match score visualisation per result
- Claude explanation of why each movie was recommended

**Tab 2 — New Movie Concept**
- Pitch an unmade movie: title + plot + genre selection
- System finds most similar existing movies
- Core cold-start demo — pure metadata, no user data

---

## Project Structure

```
S04-cold-start/
├── data/
│   ├── raw/                    # TMDB CSV (not committed — too large)
│   └── processed/
│       ├── movies_clean.csv    # 44,937 filtered movies
│       ├── embeddings.npy      # (44937, 384) float32 (not committed)
│       └── faiss_index.bin     # FAISS index (not committed)
├── notebooks/
│   ├── 01_exploration.py       # Dataset overview
│   ├── 02_cleaning.py          # Filter and construct text
│   ├── 03_embeddings.py        # Generate MiniLM embeddings
│   └── 04_build_index.py       # Build FAISS index
├── src/
│   └── recommender.py          # Core recommender logic
├── app/
│   └── app.py                  # Streamlit application
├── .env                        # ANTHROPIC_API_KEY (not committed)
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
pip install numpy pandas scikit-learn sentence-transformers faiss-cpu streamlit plotly anthropic python-dotenv ipykernel
```

**2. Download dataset**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
cd data/raw
kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies --unzip
```

**3. Build the index**
```bash
python notebooks/02_cleaning.py
python notebooks/03_embeddings.py      # ~9 min on CPU
python notebooks/04_build_index.py
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
| Dataset | TMDB · Kaggle · 1.4M movies |
| Embeddings | sentence-transformers · all-MiniLM-L6-v2 · 384-dim |
| Vector search | FAISS · IndexFlatIP · cosine similarity |
| LLM | Anthropic Claude Sonnet · query expansion + explanations |
| App | Streamlit · 2-tab interface |
| Runtime | Python 3.11 · conda |

---

## Extensibility

The pipeline is domain-agnostic. Any item catalogue with text metadata can replace the TMDB dataset:

- **E-commerce** (Sellpy, second-hand goods) — product title + description + category
- **Job listings** — role + description + required skills
- **Research papers** — title + abstract + keywords
- **Music** — artist + genre + lyrics snippet

The embedding + FAISS + LLM explanation pattern transfers directly.

---

## Author

**Ganisetty Sai Surya Sushil Kumar**  
MSc AI & Automation · University West, Sweden  
[github.com/SushilKumar6608](https://github.com/SushilKumar6608)