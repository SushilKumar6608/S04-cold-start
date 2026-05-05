#02_cleaning.py
import pandas as pd

# Load
df = pd.read_csv(r"D:\Egna projekt\S04-cold-start\data\raw\TMDB_movie_dataset_v11.csv")
print(f"Original shape: {df.shape}")

# Keep only useful columns
cols = ['id', 'title', 'overview', 'genres', 'keywords', 'tagline',
        'release_date', 'vote_average', 'vote_count', 'popularity', 'original_language']
df = df[cols]

# Drop rows with no overview (can't build embedding without it)
df = df.dropna(subset=['overview'])
df = df[df['overview'].str.strip() != '']
print(f"After dropping missing overviews: {df.shape}")

# Fill missing text fields with empty string
df['genres']   = df['genres'].fillna('')
df['keywords'] = df['keywords'].fillna('')
df['tagline']  = df['tagline'].fillna('')

# Drop rows with no title
df = df.dropna(subset=['title'])

# Keep only English movies with at least 10 votes (filters out junk)
df = df[df['original_language'] == 'en']
df = df[df['vote_count'] >= 10]
print(f"After language + vote filter: {df.shape}")

# Reset index
df = df.reset_index(drop=True)

# Build the text field we'll embed: overview + genres + keywords + tagline
df['text_for_embedding'] = (
    df['title'] + '. ' +
    df['tagline'] + '. ' +
    df['overview'] + ' ' +
    'Genres: ' + df['genres'] + '. ' +
    'Keywords: ' + df['keywords'] + '.'
)

print(f"\nFinal shape: {df.shape}")
print(f"\nSample text_for_embedding:\n{df['text_for_embedding'].iloc[0]}")

# Save cleaned data
df.to_csv(r"D:\Egna projekt\S04-cold-start\data\processed\movies_clean.csv", index=False)
print("\nSaved to data/processed/movies_clean.csv")