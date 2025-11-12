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

@st.cache_data
def load_data_for_plots(filepath="cairo_real_estate_dataset.csv"):
    """Loads the raw data and applies minimum cleaning necessary for EDA plots."""
    try:
        df = pd.read_csv(filepath) 
    except FileNotFoundError:
        st.error(f"Error: The data file {filepath} was not found. Cannot generate plots.")
        return None
        
    df['compound_name'] = df['compound_name'].fillna('Not Specified')
    df['finishing_type'] = df['finishing_type'].fillna('Unfinished')
    df = df.dropna(subset=['price_egp', 'district', 'compound_name'])
    
    binary_cols = ['has_balcony', 'has_parking', 'has_security', 'has_amenities']
    for col in binary_cols:
        if col not in df.columns:
            df[col] = 'No' 
    
    return df

def generate_plotly_plots(df):
    if df is None:
        return None, None, None, None
        
    top_5_compounds = df['compound_name'].value_counts().nlargest(5).index
    
    compound_counts = df['compound_name'].value_counts().reset_index(name='compound_count').head(10)
    compound_counts.columns = ['compound_name', 'compound_count']
    fig1 = px.bar(compound_counts, x='compound_count', y='compound_name', 
                  title='Top 10 Compounds by Number of Listings', color='compound_count',
                  color_continuous_scale='viridis', 
                  hover_data={'compound_count': True, 'compound_name': False})
    fig1.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    fig1.update_yaxes(title_text='Compound Name')
    fig1.update_xaxes(title_text='Number of Listings')
    
    avg_price = df.groupby('district')['price_egp'].mean().reset_index(name='Average Price (EGP)')
    avg_price = avg_price.sort_values(by='Average Price (EGP)', ascending=False)
    avg_price['Text Label'] = (avg_price['Average Price (EGP)'] / 1e6).round(2).astype(str) + 'M'
    
    fig2 = px.bar(avg_price, x='district', y='Average Price (EGP)', 
                  title='Average Price by District', color='Average Price (EGP)',
                  color_continuous_scale='plasma', text='Text Label')
    fig2.update_traces(textposition='outside')
    fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis={'categoryorder':'total descending'}, showlegend=False)
    fig2.update_yaxes(title_text='Average Price (EGP)')
    fig2.update_xaxes(title_text='District')

    df_focus = df[df['compound_name'].isin(top_5_compounds)].copy()
    
    vs_finish = df_focus.groupby(['compound_name','finishing_type'], as_index=False)['price_egp'].mean()
    fig3 = px.bar(vs_finish, x='compound_name', y='price_egp', color='finishing_type', 
                  barmode='group', title='Average Price By Top 5 Compounds and Finishing Type')
    fig3.update_layout(yaxis_title='Average Price (EGP)', xaxis_title='Compound Name')


    df_focus['service'] = (df_focus['has_security'].map({'Yes':1,"No":0}) +
                          df_focus['has_parking'].map({'Yes':1,"No":0}) +
                          df_focus['has_balcony'].map({'Yes':1,"No":0}) +
                          df_focus['has_amenities'].map({'Yes':1,"No":0}))
                          
    df_focus['service'] = df_focus['service'].astype(str)

    vs_service = df_focus.groupby(['compound_name', 'service'], as_index=False)['price_egp'].mean()
    fig4 = px.bar(vs_service, x='compound_name', y='price_egp', color='service', 
                  barmode='group', title='Average Price By Top 5 Compounds and Service Count')
    fig4.update_layout(yaxis_title='Average Price (EGP)', xaxis_title='Compound Name')
    fig4.update_xaxes(tickangle=45)
    
    return fig1, fig2, fig3, fig4
    
st.markdown("---")

@st.cache_resource
def load_model():
    """Loads the trained model pipeline."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please refresh.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_ui_features():
    """Loads the feature lists and defaults for the UI."""
    try:
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        return features
    except FileNotFoundError:
        st.error("UI features file not found. Please refresh.")
        return None
    except Exception as e:
        st.error(f"Error loading UI features: {e}")
        return None

model = load_model()
ui_features = load_ui_features()

st.title("🏠 Cairo Real Estate Price Predictor")
st.markdown("Use the controls in the sidebar to enter property details and predict its market price.")

if model is None or ui_features is None:
    st.warning("Model or UI features could not be loaded. The app might be restarting. Please refresh in a moment.")
else:
    st.sidebar.header("Enter Property Features")
    
    inputs = {} 

    st.sidebar.subheader("Property Sizing")
    num_features = ui_features.get('numeric_features', {})
    
    area_key = 'area_sqm'
    if area_key in num_features:
        f = num_features[area_key]
        inputs[area_key] = st.sidebar.number_input(
            "Area (sqm)", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default'],
            step=10.0
        )
        
    bed_key = 'bedrooms'
    if bed_key in num_features:
        f = num_features[bed_key]
        inputs[bed_key] = st.sidebar.slider(
            "Bedrooms", 
            min_value=int(f['min']), 
            max_value=int(f['max']), 
            value=int(f['default']),
            step=1
        )

    bath_key = 'bathrooms'
    if bath_key in num_features:
        f = num_features[bath_key]
        inputs[bath_key] = st.sidebar.slider(
            "Bathrooms", 
            min_value=int(f['min']), 
            max_value=int(f['max']), 
            value=int(f['default']),
            step=1
        )

    floor_key = 'floor_number'
    if floor_key in num_features:
        f = num_features[floor_key]
        inputs[floor_key] = st.sidebar.number_input(
            "Floor Number", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default'],
            step=1.0
        )
        
    age_key = 'building_age_years'
    if age_key in num_features:
        f = num_features[age_key]
        inputs[age_key] = st.sidebar.number_input(
            "Building Age (years)", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default'],
            step=1.0
        )

    st.sidebar.subheader("Location")
    
    dist_key = 'district'
    if dist_key in ui_features:
        inputs[dist_key] = st.sidebar.selectbox("District", ui_features[dist_key])

    comp_key = 'compound_name'
    if comp_key in ui_features:
        inputs[comp_key] = st.sidebar.selectbox("Compound", ui_features[comp_key])

    auc_key = 'distance_to_auc_km'
    if auc_key in num_features:
        f = num_features[auc_key]
        inputs[auc_key] = st.sidebar.slider(
            "Distance to AUC (km)", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default']
        )
        
    mall_key = 'distance_to_mall_km'
    if mall_key in num_features:
        f = num_features[mall_key]
        inputs[mall_key] = st.sidebar.slider(
            "Distance to Mall (km)", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default']
        )

    metro_key = 'distance_to_metro_km'
    if metro_key in num_features:
        f = num_features[metro_key]
        inputs[metro_key] = st.sidebar.slider(
            "Distance to Metro (km)", 
            min_value=f['min'], 
            max_value=f['max'], 
            value=f['default']
        )

    st.sidebar.subheader("Amenities & Finishing")
    
    finish_key = 'finishing_type'
    if finish_key in ui_features:
        inputs[finish_key] = st.sidebar.selectbox("Finishing Type", ui_features[finish_key])

    view_key = 'view_type'
    if view_key in ui_features:
        inputs[view_key] = st.sidebar.selectbox("View Type", ui_features[view_key])

    col1, col2 = st.sidebar.columns(2)
    
    balcony_key = 'has_balcony'
    if balcony_key in ui_features:
        inputs[balcony_key] = col1.selectbox("Balcony", ui_features[balcony_key], index=1)
        
    parking_key = 'has_parking'
    if parking_key in ui_features:
        inputs[parking_key] = col2.selectbox("Parking", ui_features[parking_key], index=1)
        
    security_key = 'has_security'
    if security_key in ui_features:
        inputs[security_key] = col1.selectbox("Security", ui_features[security_key], index=1)

    amenities_key = 'has_amenities'
    if amenities_key in ui_features:
        inputs[amenities_key] = col2.selectbox("Amenities", ui_features[amenities_key], index=1)


    if st.sidebar.button("Predict Price", type="primary", use_container_width=True):
        
        all_cols = list(num_features.keys()) + list(ui_features.keys())
        filtered_inputs = {k: [v] for k, v in inputs.items() if k in all_cols}
        input_df = pd.DataFrame(filtered_inputs)
        
        training_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        input_df = input_df[training_cols]

        try:
            prediction = model.predict(input_df)
            predicted_price = prediction[0]
            
            st.success(f"**Estimated Price: {predicted_price:,.0f} EGP**")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.dataframe(input_df)







