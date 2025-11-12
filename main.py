import streamlit as st
import pandas as pd
import joblib
import json
import os
import time
import numpy as np

from Model_Training import train_and_save_model
from Model_Training import NUMERIC_FEATURES, CATEGORICAL_FEATURES 

st.set_page_config(
    page_title="Cairo Real Estate Predictor",
    page_icon="🏠",
    layout="wide"
)

DATA_FILEPATH = "cairo_real_estate_dataset.csv"
MODEL_PATH = "real_estate_model.joblib"
FEATURES_PATH = "ui_features.json"


if not os.path.exists(MODEL_PATH):
    st.title("🏠 Welcome to the Real Estate Predictor")
    st.info("Performing first-time setup. This may take a few minutes.")
    with st.spinner("Training model... Please wait. This happens once on startup."):
        try:
            start_time = time.time()
            train_and_save_model(DATA_FILEPATH, MODEL_PATH, FEATURES_PATH)
            end_time = time.time()
            st.success(f"Model trained and saved! (Took {end_time - start_time:.2f} seconds)")
            st.info("App is reloading...")
            time.sleep(2)
            st.rerun() 
        except Exception as e:
            st.error(f"Error during model training: {e}")
            st.error("Please check the CSV file and app logs.")
            st.stop() 

@st.cache_resource
def load_model_and_features():
    """Loads the pre-trained model and UI features."""
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        ui_features = json.load(f)
    return model, ui_features

model, ui_features = load_model_and_features()


def run_prediction(inputs, model):

    
    all_cols = list(ui_features['numeric_features'].keys()) + list(ui_features.keys())
    
    filtered_inputs = {k: [v] for k, v in inputs.items() if k in all_cols}
    input_df = pd.DataFrame(filtered_inputs)
    

    if 'compound_count' in NUMERIC_FEATURES and 'compound_count' not in input_df.columns:

        input_df['compound_count'] = 100 
        

    training_cols = NUMERIC_FEATURES + [c for c in CATEGORICAL_FEATURES if c != 'services']
    input_df = input_df[training_cols]
    
    prediction_array = model.predict(input_df)
    
    return np.exp(prediction_array[0]) 


st.title("🏠 Cairo Real Estate Price Predictor")

num_features = ui_features.pop('numeric_features')
cat_features = ui_features

st.sidebar.header("Property Details")
inputs = {}

for col in ['area_sqm', 'bedrooms', 'bathrooms', 'floor_number']:
    if col in num_features:
        min_val = num_features[col]['min']
        max_val = num_features[col]['max']
        default_val = max(1, int(np.median([min_val, max_val])))
        inputs[col] = st.sidebar.slider(
            col.replace('_', ' ').title(),
            min_value=min_val,
            max_value=max_val,
            value=default_val
        )

st.header("Location & Age")
col1, col2, col3 = st.columns(3)
with col1:
    col = 'building_age_years'
    if col in num_features:
        min_val = num_features[col]['min']
        max_val = num_features[col]['max']
        inputs[col] = st.slider("Building Age (Years)", min_val, max_val, max(0, int(np.median([min_val, max_val]))))
with col2:
    col = 'distance_to_auc_km'
    if col in num_features:
        inputs[col] = st.number_input("Distance to AUC (km)", float(num_features[col]['min']), float(num_features[col]['max']), value=1.0, step=0.1)
with col3:
    col = 'distance_to_mall_km'
    if col in num_features:
        inputs[col] = st.number_input("Distance to Mall (km)", float(num_features[col]['min']), float(num_features[col]['max']), value=5.0, step=0.1)
    col = 'distance_to_metro_km'
    if col in num_features:
        inputs[col] = st.number_input("Distance to Metro (km)", float(num_features[col]['min']), float(num_features[col]['max']), value=10.0, step=0.1)


st.header("Property Classification")
col1, col2, col3 = st.columns(3)

if 'compound_name' in cat_features:
    inputs['compound_name'] = col1.selectbox("Compound Name", ['Not Specified'] + cat_features['compound_name'])

if 'district' in cat_features:
    inputs['district'] = col2.selectbox("District", cat_features['district'])

if 'finishing_type' in cat_features:
    inputs['finishing_type'] = col3.selectbox("Finishing Type", cat_features['finishing_type'])

if 'view_type' in cat_features:
    inputs['view_type'] = col1.selectbox("View Type", cat_features['view_type'])

st.header("Services & Amenities (New EDA Feature)")
st.info("The combined count of these services is used as a new feature: `services`.")
col1, col2, col3, col4 = st.columns(4)

has_balcony = col1.checkbox("Balcony", value=True)
has_parking = col2.checkbox("Parking", value=True)
has_security = col3.checkbox("Security", value=True)
has_amenities = col4.checkbox("Amenities", value=False)


inputs['services'] = int(has_balcony) + int(has_parking) + int(has_security) + int(has_amenities)


if st.sidebar.button("Predict Price", type="primary", use_container_width=True):
    
    missing_cols = [col for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if col not in inputs and col not in ['compound_count', 'services']]
    if missing_cols:
         st.error(f"Error: Missing required inputs: {', '.join(missing_cols)}")
         st.stop()
    
    try:
        with st.spinner("Calculating price..."):
            predicted_price = run_prediction(inputs, model)
            
            st.markdown("---")
            st.subheader("Predicted Property Price")
            
            st.metric(label="Estimated Price (EGP)", value=f"{predicted_price:,.0f}")
            st.success("Prediction complete!")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and the model is correctly trained.")
