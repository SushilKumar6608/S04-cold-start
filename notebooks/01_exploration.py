import pandas as pd

# Load the dataset
df = pd.read_csv(r"D:\Egna projekt\S04-cold-start\data\raw\TMDB_movie_dataset_v11.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSample row:\n", df.iloc[0])