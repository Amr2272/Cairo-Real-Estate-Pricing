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
    fig1 = px.bar(compound_counts, x='compound_count', y='compound_name', title='Top 10 Compounds by Number of Listings', color='compound_count', color_continuous_scale='viridis', hover_data={'compound_count': True, 'compound_name': False})
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    fig1.update_yaxes(title_text='Compound Name')
    fig1.update_xaxes(title_text='Number of Listings')
    avg_price = df.groupby('district')['price_egp'].mean().reset_index(name='Average Price (EGP)')
    avg_price = avg_price.sort_values(by='Average Price (EGP)', ascending=False)
    avg_price['Text Label'] = (avg_price['Average Price (EGP)'] / 1e6).round(2).astype(str) + 'M'
    fig2 = px.bar(avg_price, x='district', y='Average Price (EGP)', title='Average Price by District', color='Average Price (EGP)', color_continuous_scale='plasma', text='Text Label')
    fig2.update_traces(textposition='outside')
    fig2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', xaxis={'categoryorder': 'total descending'}, showlegend=False)
    fig2.update_yaxes(title_text='Average Price (EGP)')
    fig2.update_xaxes(title_text='District')
    df_focus = df[df['compound_name'].isin(top_5_compounds)].copy()
    vs_finish = df_focus.groupby(['compound_name', 'finishing_type'], as_index=False)['price_egp'].mean()
    fig3 = px.bar(vs_finish, x='compound_name', y='price_egp', color='finishing_type', barmode='group', title='Average Price By Top 5 Compounds and Finishing Type')
    fig3.update_layout(yaxis_title='Average Price (EGP)', xaxis_title='Compound Name')
    df_focus['service'] = (df_focus['has_security'].map({'Yes': 1, "No": 0}) + df_focus['has_parking'].map({'Yes': 1, "No": 0}) + df_focus['has_balcony'].map({'Yes': 1, "No": 0}) + df_focus['has_amenities'].map({'Yes': 1, "No": 0}))
    df_focus['service'] = df_focus['service'].astype(str)
    vs_service = df_focus.groupby(['compound_name', 'service'], as_index=False)['price_egp'].mean()
    fig4 = px.bar(vs_service, x='compound_name', y='price_egp', color='service', barmode='group', title='Average Price By Top 5 Compounds and Service Count')
    fig4.update_layout(yaxis_title='Average Price (EGP)', xaxis_title='Compound Name')
    fig4.update_xaxes(tickangle=45)
    return fig1, fig2, fig3, fig4

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


st.sidebar.header("Enter Property Features")
inputs = {}
num_features = ui_features.get('numeric_features', {})

for key, f in num_features.items():
    if 'area' in key:
        inputs[key] = st.sidebar.number_input(
        "Area (sqm)",
        value=f.get('default', 0.0),
        step=10.0)    
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
        st.header('Model Prediction Price')
        st.success(f"🏡 **Estimated Price: {pred:,.0f} EGP**")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.dataframe(pd.DataFrame([inputs]))


