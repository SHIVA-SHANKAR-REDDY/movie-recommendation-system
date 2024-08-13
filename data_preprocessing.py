import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Preprocessing
data = data.dropna()

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save processed data
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
