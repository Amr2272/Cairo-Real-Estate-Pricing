import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from Utilities import load_data, get_preprocessor

TARGET = 'price_egp'

NUMERIC_FEATURES = [
    'area_sqm', 'bedrooms', 'bathrooms', 'floor_number', 'building_age_years',
    'distance_to_auc_km', 'distance_to_mall_km', 'distance_to_metro_km'
]

CATEGORICAL_FEATURES = [
    'district', 'compound_name', 'finishing_type', 'has_balcony',
    'has_parking', 'has_security', 'has_amenities', 'view_type'
]

def train_and_save_model(data_path, model_path, features_path):

    print(f"Starting model training... Loading data from {data_path}")
    df = load_data(data_path)
    if df is None:
        raise FileNotFoundError(f"Data file not found at {data_path}")

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    for col in all_features + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Error: Column '{col}' not found in the loaded data.")

    print("Data loaded. Starting preprocessing and training...")

    X = df[all_features]
    y = df[TARGET]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = get_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])


    best_params = {
        'model__learning_rate': 0.05,
        'model__max_depth': 10,
        'model__n_estimators': 300
    }
    
    final_model = pipeline
    final_model.set_params(**best_params)
    
    print("Training final model on all data...")
    final_model.fit(X, y)
    
    print("Model training complete.")

    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    ui_features = {}
    for col in CATEGORICAL_FEATURES:
        ui_features[col] = sorted(list(X[col].unique().astype(str)))
    
    ui_features['numeric_features'] = {}
    for col in NUMERIC_FEATURES:
        ui_features['numeric_features'][col] = {
            'min': float(X[col].min()),
            'max': float(X[col].max()),
            'default': float(X[col].median())
        }

    with open(features_path, 'w') as f:
        json.dump(ui_features, f, indent=4)
    print(f"UI feature definitions saved to {features_path}")
    
    return True

