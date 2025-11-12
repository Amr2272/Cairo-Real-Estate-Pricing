import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from Utilties import SCALED_COLS, FINISHING_MAP, BINARY_MAP, LABEL_ENCODED_COLS

MODEL_FILE = 'best_gbr_model.joblib'
SCALER_FILE = 'scaler.joblib'
ENCODERS_FILE = 'encoders.joblib'

TRAINING_COLUMNS = [
    'area_sqm', 'bedrooms', 'bathrooms', 'floor_number', 'building_age_years',
    'district', 'compound_name', 'distance_to_auc_km', 'distance_to_mall_km',
    'distance_to_metro_km', 'finishing_type', 'has_balcony', 'has_parking',
    'has_security', 'has_amenities', 'view_type', 'days_on_market',
    'seller_type', 'is_negotiable', 'services'
]

@st.cache_resource
def load_resources():
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE) or not os.path.exists(ENCODERS_FILE):
        st.error("Model resources not found. Please run 'python train.py' first.")
        return None, None, None
        
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        lb_dict = joblib.load(ENCODERS_FILE)
        return model, scaler, lb_dict
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None, None

model, scaler, lb_dict = load_resources()

st.set_page_config(page_title="Cairo Real Estate Price Predictor", layout="wide")
st.title("🏡 Cairo Real Estate Price Predictor")
st.markdown("Use this tool to estimate the sale price of a property in the New Cairo area.")

if model and scaler and lb_dict:
    district_options = list(lb_dict['district'].classes_)
    compound_options = list(lb_dict['compound_name'].classes_)
    view_options = list(lb_dict['view_type'].classes_)

    with st.sidebar:
        st.header("Core Property Details")
        area_sqm = st.slider("Area (sqm)", min_value=50, max_value=500, value=150, step=5)
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
        bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
        floor_number = st.slider("Floor Number", min_value=1, max_value=20, value=5)
        building_age_years = st.slider("Building Age (years)", min_value=0, max_value=30, value=5)
        days_on_market = st.slider("Days on Market", min_value=1, max_value=365, value=90)
        
        seller_type_label = st.selectbox("Seller Type", options=['Broker', 'Owner'], index=0)
        is_negotiable_label = st.selectbox("Is Negotiable", options=['Yes', 'No'], index=0)

    st.subheader("Location and Finishing")
    col1, col2 = st.columns(2)
    with col1:
        district_label = st.selectbox("District", options=district_options)
        compound_name_label = st.selectbox("Compound Name", options=compound_options)
        view_type_label = st.selectbox("View Type", options=view_options)
        finishing_type_label = st.selectbox("Finishing Type", options=list(FINISHING_MAP.keys()))
        
    with col2:
        distance_to_auc_km = st.number_input("Dist. to AUC (km)", min_value=0.1, max_value=50.0, value=10.0, step=0.1)
        distance_to_mall_km = st.number_input("Dist. to Mall (km)", min_value=0.1, max_value=20.0, value=5.0, step=0.1)
        distance_to_metro_km = st.number_input("Dist. to Metro (km)", min_value=0.1, max_value=20.0, value=10.0, step=0.1)

    st.subheader("Amenities")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        has_balcony_label = st.selectbox("Has Balcony", options=['Yes', 'No'], index=0)
    with col4:
        has_parking_label = st.selectbox("Has Parking", options=['Yes', 'No'], index=0)
    with col5:
        has_security_label = st.selectbox("Has Security", options=['Yes', 'No'], index=0)
    with col6:
        has_amenities_label = st.selectbox("Has Amenities", options=['Yes', 'No'], index=0)


    if st.button("Predict Price", type="primary"):
        input_data = {
            'area_sqm': area_sqm,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floor_number': floor_number,
            'building_age_years': building_age_years,
            'distance_to_auc_km': distance_to_auc_km,
            'distance_to_mall_km': distance_to_mall_km,
            'distance_to_metro_km': distance_to_metro_km,
            'days_on_market': days_on_market,

            'finishing_type': FINISHING_MAP[finishing_type_label],
            'has_balcony': BINARY_MAP[has_balcony_label],
            'has_parking': BINARY_MAP[has_parking_label],
            'has_security': BINARY_MAP[has_security_label],
            'has_amenities': BINARY_MAP[has_amenities_label],
            'seller_type': BINARY_MAP[seller_type_label],
            'is_negotiable': BINARY_MAP[is_negotiable_label],
            
            'district': lb_dict['district'].transform([district_label])[0],
            'compound_name': lb_dict['compound_name'].transform([compound_name_label])[0],
            'view_type': lb_dict['view_type'].transform([view_type_label])[0],
        }

        input_data['services'] = (input_data['has_security'] + input_data['has_parking'] +
                                  input_data['has_balcony'] + input_data['has_amenities'])

        input_df = pd.DataFrame([input_data])
        input_df = input_df[TRAINING_COLUMNS] 

        input_df[SCALED_COLS] = scaler.transform(input_df[SCALED_COLS])

        try:
            prediction = model.predict(input_df)[0]
            st.success(f"## Predicted Price: EGP {prediction:,.0f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.warning("Prediction failed. Check input values and model integrity.")
            
else:
    st.warning("Model is not loaded. Please ensure all model files ('best_gbr_model.joblib', 'scaler.joblib', 'encoders.joblib') are present and compatible.")
