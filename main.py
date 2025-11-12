import streamlit as st
import pandas as pd
import joblib
import json
import os
import time
import numpy as np
import plotly.express as px
from Model_Training import train_and_save_model
from Model_Training import NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(page_title="Cairo Real Estate Predictor", page_icon="🏠", layout="wide")

DATA_FILEPATH = "cairo_real_estate_dataset.csv"
MODEL_PATH = "real_estate_model.joblib"
FEATURES_PATH = "ui_features.json"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(FEATURES_PATH, "r") as f:
                ui_features = json.load(f)
            return model, ui_features
        except Exception as e:
            st.error(f"Error loading model or features: {e}")
            st.stop()
    else:
        st.warning("Model or features not found. Training a new one — this happens once.")
        with st.spinner("Training model... please wait"):
            try:
                start = time.time()
                train_and_save_model(DATA_FILEPATH, MODEL_PATH, FEATURES_PATH)
                duration = time.time() - start
                st.success(f"Model trained successfully in {duration:.2f} seconds.")
                model = joblib.load(MODEL_PATH)
                with open(FEATURES_PATH, "r") as f:
                    ui_features = json.load(f)
                return model, ui_features
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

model, ui_features = load_or_train_model()

@st.cache_data
def load_data_for_plots(filepath=DATA_FILEPATH):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Data file '{filepath}' not found.")
        return None
    df['compound_name'] = df['compound_name'].fillna('Not Specified')
    df['finishing_type'] = df['finishing_type'].fillna('Unfinished')
    df = df.dropna(subset=['price_egp', 'district', 'compound_name'])
    binary_cols = ['has_balcony', 'has_parking', 'has_security', 'has_amenities']
    for col in binary_cols:
        if col not in df.columns:
            df[col] = 'No'
    return df

st.title("🏠 Cairo Real Estate Price Predictor")
st.markdown("Use the sidebar to enter property details and predict the market price.")

if model is None or ui_features is None:
    st.error("Model or UI features could not be loaded. Please refresh later.")
    st.stop()

st.sidebar.header("Enter Property Features")
inputs = {}
num_features = ui_features.get('numeric_features', {})

for key, f in num_features.items():
    if 'area' in key:
        inputs[key] = st.sidebar.number_input(f"Area (sqm)", min_value=f['min'], max_value=f['max'], value=f['default'], step=10.0)
    elif key in ['bedrooms', 'bathrooms']:
        inputs[key] = st.sidebar.slider(key.capitalize(), int(f['min']), int(f['max']), int(f['default']), 1)
    else:
        inputs[key] = st.sidebar.number_input(key.replace('_', ' ').capitalize(), min_value=f['min'], max_value=f['max'], value=f['default'], step=1.0)

for key in ui_features.keys():
    if key not in num_features:
        inputs[key] = st.sidebar.selectbox(key.replace('_', ' ').capitalize(), ui_features[key])

if st.sidebar.button("Predict Price", type="primary", use_container_width=True):
    try:
        all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        input_df = pd.DataFrame([{k: inputs[k] for k in all_cols if k in inputs}])
        pred = model.predict(input_df)[0]
        st.success(f"🏡 **Estimated Price: {pred:,.0f} EGP**")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.dataframe(pd.DataFrame([inputs]))

st.markdown("---")
st.header("📊 Market Insights")

df = load_data_for_plots()
if df is not None:
    with st.spinner("Generating charts..."):
        top_compounds = df['compound_name'].value_counts().nlargest(5).index
        fig1 = px.bar(
            df['compound_name'].value_counts().head(10).reset_index(),
            x='count', y='compound_name',
            title="Top 10 Compounds by Listings", color='count', color_continuous_scale='viridis'
        )
        fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        avg_price = df.groupby('district')['price_egp'].mean().reset_index().sort_values('price_egp', ascending=False)
        fig2 = px.bar(avg_price, x='district', y='price_egp', title="Average Price by District", color='price_egp', color_continuous_scale='plasma')
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Data unavailable — upload 'cairo_real_estate_dataset.csv' to enable insights.")
