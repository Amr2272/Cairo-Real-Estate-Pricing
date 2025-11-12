import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
from Utilities import load_data, preprocess_for_training, get_feature_order

warnings.filterwarnings('ignore')

def train_and_save_model():


    print("Loading data...")
    df = load_data('cairo_real_estate_dataset.csv')
    
    print("Preprocessing data for training...")

    compound_count_map = df.groupby('compound_name')['listing_id'].count().to_dict()
    
    x_train, x_test, y_train, y_test, scaler, encoders, _ = preprocess_for_training(df.copy())
    
    feature_cols = get_feature_order(df.copy())

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(x_train, y_train)

    print("Evaluating model...")
    pred = model.predict(x_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    
    print(f"Model Performance - R2 Score: {r2:.4f}, MAE: {mae:,.0f}")

    print("Saving artifacts...")
    joblib.dump(model, 'real_estate_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(encoders, 'label_encoders.joblib')
    joblib.dump(compound_count_map, 'compound_map.joblib')
    joblib.dump(feature_cols, 'feature_columns.joblib')
    
    print("Model, scaler, encoders, compound map, and feature columns saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
