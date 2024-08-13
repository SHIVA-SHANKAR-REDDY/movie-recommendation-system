import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier

# Load processed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Feature selection
X_train = train_data[['userId', 'movieId']]
y_train = train_data['rating']
X_test = test_data[['userId', 'movieId']]
y_test = test_data['rating']

# Model definition
model = KNeighborsClassifier()

# Hyperparameter tuning
param_grid = {'n_neighbors': [5, 10, 15]}
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

# Evaluate model
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f'RMSE: {rmse}')

# Save the model
import joblib
joblib.dump(best_model, 'movie_recommender_model.pkl')
