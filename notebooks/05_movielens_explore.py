#05_movielens_explore.py
import pandas as pd

RAW = r"D:\Egna projekt\S04-cold-start\data\raw\ml-25m"

# Movies
movies = pd.read_csv(f"{RAW}/movies.csv")
print("=== movies.csv ===")
print(f"Shape: {movies.shape}")
print(movies.head(3))

# Ratings
ratings = pd.read_csv(f"{RAW}/ratings.csv")
print("\n=== ratings.csv ===")
print(f"Shape: {ratings.shape}")
print(ratings.head(3))
print(f"Unique users: {ratings['userId'].nunique()}")
print(f"Unique movies: {ratings['movieId'].nunique()}")
print(f"Rating range: {ratings['rating'].min()} – {ratings['rating'].max()}")

# Links
links = pd.read_csv(f"{RAW}/links.csv")
print("\n=== links.csv ===")
print(f"Shape: {links.shape}")
print(links.head(3))

# Extract release year from title
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
print("\n=== Release year distribution ===")
print(movies['year'].describe())
print(f"\nMovies post-2015: {(movies['year'] > 2015).sum()}")
print(f"Movies pre-2015:  {(movies['year'] <= 2015).sum()}")

# How many ratings do post-2015 movies get?
post2015_ids = movies[movies['year'] > 2015]['movieId']
post2015_ratings = ratings[ratings['movieId'].isin(post2015_ids)]
print(f"\nRatings for post-2015 movies: {len(post2015_ratings)}")
print(f"Avg ratings per post-2015 movie: {len(post2015_ratings) / len(post2015_ids):.1f}")

pre2015_ids = movies[movies['year'] <= 2015]['movieId']
pre2015_ratings = ratings[ratings['movieId'].isin(pre2015_ids)]
print(f"Avg ratings per pre-2015 movie:  {len(pre2015_ratings) / len(pre2015_ids):.1f}")