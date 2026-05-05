import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import faiss

PROCESSED = r"D:\Egna projekt\S04-cold-start\data\processed"
PLOTS     = r"D:\Egna projekt\S04-cold-start\docs"
RAW       = r"D:\Egna projekt\S04-cold-start\data\raw\ml-25m"

results_df = pd.read_csv(f"{PROCESSED}/coldstart_evaluation.csv")
tmdb_clean = pd.read_csv(f"{PROCESSED}/movies_clean.csv")
embeddings = np.load(f"{PROCESSED}/embeddings.npy")
index      = faiss.read_index(f"{PROCESSED}/faiss_index.bin")
movies     = pd.read_csv(f"{RAW}/movies.csv")
ratings    = pd.read_csv(f"{RAW}/ratings.csv")

tmdb_clean['id'] = tmdb_clean['id'].astype(str)
tmdb_id_to_row   = {row['id']: i for i, row in tmdb_clean.iterrows()}
movies['year']   = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

S = {
    'coral':  '#E05252',
    'green':  '#2EAA7F',
    'purple': '#7B5EA7',
    'blue':   '#3A7FBF',
    'amber':  '#D4860A',
    'muted':  '#6B7280',
    'border': '#D1D5DB',
    'text':   '#111827',
    'sub':    '#6B7280',
}

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.edgecolor':   S['border'],
    'axes.labelcolor':  S['text'],
    'xtick.color':      S['muted'],
    'ytick.color':      S['muted'],
    'text.color':       S['text'],
    'grid.color':       S['border'],
    'grid.linestyle':   '--',
    'grid.alpha':       0.7,
    'font.family':      'Segoe UI',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# ── Plot 1: Coverage ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

categories   = ['Old Movies\n(pre-2015)', 'New Movies\n(post-2015, cold-start)']
cf_coverage  = [100, 0]
emb_coverage = [100, 100]
x = np.arange(len(categories))
w = 0.35

bars1 = ax.bar(x - w/2, cf_coverage,  w, label='Collaborative Filter (SVD)',
               color=S['coral'],  alpha=0.88, zorder=3)
bars2 = ax.bar(x + w/2, emb_coverage, w, label='Embedding System (ours)',
               color=S['green'], alpha=0.88, zorder=3)

for bar, val, col in zip(bars1, cf_coverage, [S['coral'], S['coral']]):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5 if val > 0 else 4,
            f'{val}%', ha='center', va='bottom',
            color=col, fontsize=13, fontweight='bold')
for bar, val in zip(bars2, emb_coverage):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom',
            color=S['green'], fontsize=13, fontweight='bold')

ax.set_ylabel('% of movies with recommendations', fontsize=11)
ax.set_title('Cold-Start Coverage: CF vs Embedding System',
             fontsize=13, fontweight='bold', pad=15, color=S['text'])
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 120)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=10, framealpha=0.9, edgecolor=S['border'])

ax.annotate('Cold-Start Zone:\nCF fails completely',
            xy=(1 - w/2, 3), xytext=(0.62, 45),
            color=S['coral'], fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=S['coral'], lw=1.5),
            ha='center')

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_coldstart_coverage.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: plot_coldstart_coverage.png")

# ── Plot 2: Rating scarcity ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

bins   = [0, 5, 20, 50, 100, 200, 500, 10000]
labels = ['0–5', '6–20', '21–50', '51–100', '101–200', '201–500', '500+']

new_ml_ids = set(movies[movies['year'] > 2015]['movieId'])
new_rating_counts = ratings[ratings['movieId'].isin(new_ml_ids)]\
    .groupby('movieId').size().reset_index(name='count')

binned = pd.cut(new_rating_counts['count'], bins=bins, labels=labels)
counts = binned.value_counts().reindex(labels)

colors = [S['coral'] if i < 3 else S['amber'] if i < 4 else S['green']
          for i in range(len(labels))]
bars = ax.bar(labels, counts.values, color=colors, alpha=0.85, zorder=3,
              edgecolor='white', linewidth=0.5)

ax.axvline(x=2.5, color=S['coral'], linewidth=1.5, linestyle='--', zorder=4)
ax.axvline(x=3.5, color=S['amber'], linewidth=1.5, linestyle='--', zorder=4)

ax.text(1.0, counts.max() * 0.90, 'Too sparse\nfor CF',
        color=S['coral'], fontsize=9, fontweight='bold', ha='center')
ax.text(3.0, counts.max() * 0.90, 'Marginal',
        color=S['amber'], fontsize=9, fontweight='bold', ha='center')
ax.text(5.0, counts.max() * 0.90, 'Usable for CF',
        color=S['green'], fontsize=9, fontweight='bold', ha='center')

for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 20,
                str(int(h)), ha='center', va='bottom',
                color=S['muted'], fontsize=9)

ax.set_xlabel('Number of ratings', fontsize=11)
ax.set_ylabel('Number of movies', fontsize=11)
ax.set_title('Rating Scarcity for New Movies (post-2015)\n'
             '87% have fewer than 50 ratings — CF cannot work reliably',
             fontsize=12, fontweight='bold', pad=15)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_ratings_scarcity.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: plot_ratings_scarcity.png")

# ── Plot 3: Embedding example ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
ax.set_facecolor('white')
ax.axis('off')

sample = results_df[results_df['ml_ratings_count'] < 50].iloc[0]
query_title = sample['title']

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

        # Header
        ax.text(0.5, 0.97,
                f'Cold-Start Movie: "{query_title}"',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=13, fontweight='bold', color=S['purple'])
        ax.text(0.5, 0.87,
                f'Only {int(sample["ml_ratings_count"])} ratings — CF cannot generate recommendations',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color=S['coral'])
        ax.text(0.5, 0.78,
                'Embedding system finds similar movies using metadata only:',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, color=S['muted'])

        # Divider
        ax.plot([0.05, 0.95], [0.72, 0.72],
                   color=S['border'], linewidth=1,
                   transform=ax.transAxes)

        for i, (title, score) in enumerate(neighbours):
            y = 0.62 - i * 0.11
            # Rank number
            ax.text(0.04, y, f'{i+1}.',
                    transform=ax.transAxes, fontsize=11,
                    color=S['muted'], va='center', fontweight='bold')
            # Title
            ax.text(0.09, y, title[:48],
                    transform=ax.transAxes, fontsize=10,
                    color=S['text'], va='center')
            # Bar
            bar_x = 0.62
            bar_w = score * 0.32
            rect = plt.Rectangle((bar_x, y - 0.032), bar_w, 0.064,
                                  transform=ax.transAxes,
                                  color=S['green'], alpha=0.75,
                                  clip_on=False)
            ax.add_patch(rect)
            ax.text(bar_x + bar_w + 0.01, y, f'{score:.2f}',
                    transform=ax.transAxes, fontsize=9,
                    color=S['green'], va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{PLOTS}/plot_embedding_example.png", dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: plot_embedding_example.png")

print("\nAll plots saved.")