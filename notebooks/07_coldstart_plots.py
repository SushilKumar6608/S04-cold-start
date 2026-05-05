#07_coldstart_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import faiss
from sentence_transformers import SentenceTransformer

PROCESSED = r"D:\Egna projekt\S04-cold-start\data\processed"
PLOTS     = r"D:\Egna projekt\S04-cold-start\docs"

results_df = pd.read_csv(f"{PROCESSED}/coldstart_evaluation.csv")
tmdb_clean = pd.read_csv(f"{PROCESSED}/movies_clean.csv")
embeddings = np.load(f"{PROCESSED}/embeddings.npy")
index      = faiss.read_index(f"{PROCESSED}/faiss_index.bin")

tmdb_clean['id']  = tmdb_clean['id'].astype(str)
tmdb_id_to_row    = {row['id']: i for i, row in tmdb_clean.iterrows()}

RAW = r"D:\Egna projekt\S04-cold-start\data\raw\ml-25m"
movies  = pd.read_csv(f"{RAW}/movies.csv")
ratings = pd.read_csv(f"{RAW}/ratings.csv")
links   = pd.read_csv(f"{RAW}/links.csv")

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

STYLE = {
    'bg':      '#0f1117',
    'panel':   '#1a1f2e',
    'border':  '#2a3a4a',
    'text':    '#e8eaf0',
    'muted':   '#6b7a8d',
    'purple':  '#b39ddb',
    'green':   '#00d4aa',
    'blue':    '#5ba3f5',
    'amber':   '#f59e0b',
    'coral':   '#ff8a80',
}

plt.rcParams.update({
    'figure.facecolor':  STYLE['bg'],
    'axes.facecolor':    STYLE['panel'],
    'axes.edgecolor':    STYLE['border'],
    'axes.labelcolor':   STYLE['text'],
    'xtick.color':       STYLE['muted'],
    'ytick.color':       STYLE['muted'],
    'text.color':        STYLE['text'],
    'grid.color':        STYLE['border'],
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'Segoe UI',
})

# ── Plot 1: CF coverage vs Embedding coverage ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=STYLE['bg'])
ax.set_facecolor(STYLE['panel'])

categories = ['Old Movies\n(pre-2015)', 'New Movies\n(post-2015, cold-start)']
cf_coverage  = [100, 0]
emb_coverage = [100, 100]
x = np.arange(len(categories))
w = 0.35

bars1 = ax.bar(x - w/2, cf_coverage,  w, label='Collaborative Filter (SVD)',
               color=STYLE['coral'],  alpha=0.85, zorder=3)
bars2 = ax.bar(x + w/2, emb_coverage, w, label='Embedding System (ours)',
               color=STYLE['green'], alpha=0.85, zorder=3)

for bar, val in zip(bars1, cf_coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom',
            color=STYLE['coral'], fontsize=13, fontweight='bold')
for bar, val in zip(bars2, emb_coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom',
            color=STYLE['green'], fontsize=13, fontweight='bold')

ax.set_ylabel('% of movies with recommendations', color=STYLE['text'], fontsize=11)
ax.set_title('Cold-Start Coverage: CF vs Embedding System',
             color=STYLE['text'], fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 120)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(facecolor=STYLE['panel'], edgecolor=STYLE['border'],
          labelcolor=STYLE['text'], fontsize=10)

# Cold-start zone annotation
ax.annotate('Cold-Start Zone\nCF fails completely',
            xy=(1 - w/2, 5), xytext=(1 - w/2 - 0.25, 40),
            color=STYLE['coral'], fontsize=9,
            arrowprops=dict(arrowstyle='->', color=STYLE['coral']),
            ha='center')

for spine in ax.spines.values():
    spine.set_edgecolor(STYLE['border'])

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_coldstart_coverage.png", dpi=150,
            bbox_inches='tight', facecolor=STYLE['bg'])
plt.close()
print("Saved: plot_coldstart_coverage.png")

# ── Plot 2: Ratings scarcity for new movies ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=STYLE['bg'])
ax.set_facecolor(STYLE['panel'])

bins = [0, 5, 20, 50, 100, 200, 500, 10000]
labels = ['0–5', '6–20', '21–50', '51–100', '101–200', '201–500', '500+']

links['tmdbId'] = links['tmdbId'].fillna(0).astype(int).astype(str)
new_ml_ids = set(movies[movies['year'] > 2015]['movieId'])
new_rating_counts = ratings[ratings['movieId'].isin(new_ml_ids)]\
    .groupby('movieId').size().reset_index(name='count')

binned = pd.cut(new_rating_counts['count'], bins=bins, labels=labels)
counts = binned.value_counts().reindex(labels)

bars = ax.bar(labels, counts.values, color=STYLE['purple'], alpha=0.85, zorder=3)

# Highlight the "too few for CF" zone
ax.axvline(x=2.5, color=STYLE['coral'], linewidth=1.5, linestyle='--', zorder=4)
ax.text(1.0, counts.max() * 0.85, 'Too sparse\nfor CF',
        color=STYLE['coral'], fontsize=9, ha='center')
ax.text(4.5, counts.max() * 0.85, 'Marginally\nusable for CF',
        color=STYLE['amber'], fontsize=9, ha='center')

for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 5,
                str(int(h)), ha='center', va='bottom',
                color=STYLE['muted'], fontsize=9)

ax.set_xlabel('Number of ratings', color=STYLE['text'], fontsize=11)
ax.set_ylabel('Number of movies', color=STYLE['text'], fontsize=11)
ax.set_title('Rating Scarcity for New Movies (post-2015)\nCF needs hundreds of ratings to work well',
             color=STYLE['text'], fontsize=13, fontweight='bold', pad=15)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(STYLE['border'])

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_ratings_scarcity.png", dpi=150,
            bbox_inches='tight', facecolor=STYLE['bg'])
plt.close()
print("Saved: plot_ratings_scarcity.png")

# ── Plot 3: Example embedding neighbours for a cold-start movie ────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=STYLE['bg'])
ax.set_facecolor(STYLE['panel'])
ax.axis('off')

# Pick a recognisable new movie from results
sample = results_df[results_df['ml_ratings_count'] < 50].iloc[0]
query_title = sample['title']

# Find it in TMDB
match = tmdb_clean[tmdb_clean['title'].str.contains(
    query_title.split(':')[0].strip(), case=False, na=False)]

if len(match) > 0:
    row_idx = tmdb_id_to_row.get(str(match.iloc[0]['id']))
    if row_idx is not None:
        q = embeddings[row_idx:row_idx+1].copy()
        faiss.normalize_L2(q)
        scores, indices = index.search(q, 6)
        neighbours = [(tmdb_clean.iloc[i]['title'], float(s))
                      for i, s in zip(indices[0][1:], scores[0][1:])]

        ax.text(0.5, 0.95, f'Cold-Start Movie: "{query_title}"',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=12, fontweight='bold', color=STYLE['purple'])
        ax.text(0.5, 0.87, f'Only {int(sample["ml_ratings_count"])} ratings → CF cannot help',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color=STYLE['coral'])
        ax.text(0.5, 0.79, 'Embedding system finds similar movies using metadata only:',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color=STYLE['muted'])

        for i, (title, score) in enumerate(neighbours):
            y = 0.68 - i * 0.12
            ax.text(0.08, y, f'{i+1}. {title[:50]}',
                    transform=ax.transAxes, fontsize=10,
                    color=STYLE['text'], va='center')
            bar_x, bar_w = 0.60, score * 0.35
            ax.barh(y, bar_w, height=0.07, left=bar_x,
                    transform=ax.transAxes, color=STYLE['green'],
                    alpha=0.7)
            ax.text(bar_x + bar_w + 0.01, y, f'{score:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    color=STYLE['green'], va='center')

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_embedding_example.png", dpi=150,
            bbox_inches='tight', facecolor=STYLE['bg'])
plt.close()
print("Saved: plot_embedding_example.png")

print("\nAll plots saved to docs/")