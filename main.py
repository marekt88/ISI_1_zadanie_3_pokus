# main.py

# Project Description:
# This script is designed to solve a regression task using Ridge regression.
# It loads training data from X_public.npy and y_public.npy, trains a Ridge regression model,
# evaluates the model using R² score, and makes predictions on X_eval.npy.

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split

# Load training data with allow_pickle=True
X_public = np.load("X_public.npy", allow_pickle=True)
y_public = np.load("y_public.npy", allow_pickle=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_public, y_public, test_size=0.2, random_state=42)

print("Data shapes:")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Identify categorical columns (assuming first 10 columns are categorical based on the output)
categorical_features = list(range(10))
numeric_features = list(range(10, X_public.shape[1]))

# Create a column transformer with one-hot encoding for categorical features and preprocessing for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor, feature selection, and Ridge model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(Ridge(alpha=1.0), threshold='median')),
    ('ridge', Ridge())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'ridge__alpha': np.logspace(-2, 4, 20),
    'feature_selection__estimator__alpha': [0.1, 1.0, 10.0]
}

# Perform grid search with cross-validation
print("\nStarting grid search...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best cross-validation R² score:", grid_search.best_score_)

# Use the best model for final predictions
best_model = grid_search.best_estimator_

# Evaluate the model using R² score on training data
y_train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
print("\nR² score on training data:", train_r2)

# Evaluate the model using R² score on test data
y_test_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print("R² score on test data:", test_r2)

# Calculate overall R² score on all public data
y_public_pred = best_model.predict(X_public)
overall_r2 = r2_score(y_public, y_public_pred)
print("Overall R² score on all data:", overall_r2)

# Load evaluation data and make predictions
X_eval = np.load("X_eval.npy", allow_pickle=True)
y_predikcia = best_model.predict(X_eval)
np.save("y_predikcia.npy", y_predikcia)
print("\nPredictions saved to y_predikcia.npy.")
