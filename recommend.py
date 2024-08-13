import joblib
import pandas as pd

# Load the trained model
model = joblib.load('movie_recommender_model.pkl')

# Sample user input
user_id = 1
movie_id = 500

# Generate recommendation
prediction = model.predict([[user_id, movie_id]])
print(f'Predicted rating for user {user_id} on movie {movie_id}: {prediction[0]}')
