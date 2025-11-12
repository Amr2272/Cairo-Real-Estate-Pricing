import streamlit as st
import pandas as pd
import numpy as np
from utility import train_model, SCALED_COLS, FINISHING_MAP

DATA_FILE = 'cairo_real_estate_dataset.csv'

TRAINING_COLUMNS = [
    'area_sqm', 'bedrooms', 'bathrooms', 'floor_number', 'building_age_years',
    'district', 'compound_name', 'distance_to_auc_km', 'distance_to_mall_km',
    'distance_to_metro_km', 'finishing_type', 'has_balcony', 'has_parking',
    'has_security', 'has_amenities', 'view_type', 'days_on_market',
    'seller_type', 'is_negotiable', 'services'
]


st.set_page_config(page_title="Cairo Real Estate Price Predictor", layout="wide")
st.title("🏡 Cairo Real Estate Price Predictor")
st.markdown("Estimate the sale price of a property in New Cairo by adjusting the features below.")

@st.cache_resource
def get_trained_resources():
    """Loads data, trains the model, and returns resources."""
    st.info("Loading and training model... This only happens on the first run after deployment.")
    try:
        model, scaler, lb_dict = train_model(DATA_FILE)
        st.success("Model trained and ready! Start predicting.")
        return model, scaler, lb_dict
    except FileNotFoundError:
        st.error(f"Error: Data file '{DATA_FILE}' not found. Please ensure it is uploaded.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        return None, None, None

model, scaler, lb_dict = get_trained_resources()

if model:
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
            'has_balcony': lb_dict['has_balcony'].transform([has_balcony_label])[0],
            'has_parking': lb_dict['has_parking'].transform([has_parking_label])[0],
            'has_security': lb_dict['has_security'].transform([has_security_label])[0],
            'has_amenities': lb_dict['has_amenities'].transform([has_amenities_label])[0],
            'seller_type': lb_dict['seller_type'].transform([seller_type_label])[0],
            'is_negotiable': lb_dict['is_negotiable'].transform([is_negotiable_label])[0],
            
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
            
else:
    st.warning("Application is not running because essential resources could not be loaded or trained.")
