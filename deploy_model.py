import boto3
import joblib

# Load the trained model
model = joblib.load('movie_recommender_model.pkl')

# AWS S3 setup
s3 = boto3.client('s3')
bucket_name = 'your-bucket-name'

# Save model to S3
with open('movie_recommender_model.pkl', 'rb') as data:
    s3.upload_fileobj(data, bucket_name, 'movie_recommender_model.pkl')

print('Model deployed to AWS S3')
