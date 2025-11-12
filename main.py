import streamlit as st
import pandas as pd
import joblib
import json
import os
import time
import numpy as np
import plotly.express as px
from Utilities import load_data 

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

@st.cache_resource
def load_model_and_features():
    """Loads the pre-trained model and UI features."""
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        ui_features = json.load(f)
    return model, ui_features

model, ui_features = load_model_and_features()


st.header("Exploratory Data Analysis (EDA) Insights")

df_plots = load_data_for_plots(DATA_FILEPATH)
fig1, fig2, fig3, fig4 = generate_plotly_plots(df_plots)

if fig1 is not None:
    st.subheader("1. Top 10 Compounds by Number of Listings")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Average Price by District")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Average Price By Top 5 Compounds and Finishing Type")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Average Price By Top 5 Compounds and Service Count")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")


def run_prediction(inputs, model):

    model_features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    filtered_inputs = {k: [v] for k, v in inputs.items() if k in model_features}
    
    input_df = pd.DataFrame(filtered_inputs)
    

    if 'compound_count' in NUMERIC_FEATURES and 'compound_count' not in input_df.columns:
        input_df['compound_count'] = 100 
        

    training_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    for col in training_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan 

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
