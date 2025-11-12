import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from Utilities import preprocess_and_split, evaluate_model

DATA_FILE = 'cairo_real_estate_dataset.csv'
MODEL_FILE = 'best_gbr_model.joblib'
SCALER_FILE = 'scaler.joblib'
ENCODERS_FILE = 'encoders.joblib'

def train_best_model():

    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please ensure it's in the same directory.")
        return

    x_train, x_test, y_train, y_test, scaler, lb_dict = preprocess_and_split(df)
    
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(lb_dict, ENCODERS_FILE)
    print(f"Scaler saved to {SCALER_FILE}")
    print(f"LabelEncoders saved to {ENCODERS_FILE}")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'learning_rate': [0.01, 0.05, 0.1]
    }

    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )

    print("\nStarting Grid Search for GradientBoostingRegressor...")
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    joblib.dump(best_model, MODEL_FILE)

    r2, mae = evaluate_model(best_model, x_test, y_test)
    print(f"\n--- Training Complete ---")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Cross-Validation MAE: {(-grid_search.best_score_):,.0f} EGP")
    print(f"Test R2 Score: {r2:.4f}")
    print(f"Test MAE: {mae:,.0f} EGP")
    print(f"Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_best_model()
