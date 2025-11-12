import streamlit as st
import pandas as pd
import joblib
from Utilities import load_data, get_column_lists

st.set_page_config(
    page_title="Cairo Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_resources():

    try:
        model = joblib.load('real_estate_model.joblib')
        scaler = joblib.load('scaler.joblib')
        encoders = joblib.load('label_encoders.joblib')
        compound_map = joblib.load('compound_map.joblib')
        feature_cols = joblib.load('feature_columns.joblib')
        
        df_raw = load_data('cairo_real_estate_dataset.csv')
        df_raw['compound_count'] = df_raw['compound_name'].map(compound_map).fillna(df_raw['compound_name'].mode().map(compound_map).iloc[0])


        l_cols, ht_cols, n_cols, other_num_cols, f_types = get_column_lists()
        
        dropdown_options = {}
        for col in l_cols + ht_cols:

            dropdown_options[col] = df_raw[col].unique()
        dropdown_options['finishing_type'] = f_types
        
        return model, scaler, encoders, compound_map, feature_cols, df_raw, l_cols, ht_cols, n_cols, other_num_cols, dropdown_options
    except FileNotFoundError:
        st.error("Model files not found. Please run `Model Training.py` first to generate the necessary files (e.g., `real_estate_model.joblib`).")
        return (None,) * 11

(model, scaler, encoders, compound_map, 
 feature_cols, df_raw, l_cols, ht_cols, 
 n_cols, other_num_cols, dropdown_options) = load_resources()

st.title("🏠 Cairo Real Estate Price Predictor")
st.markdown("Enter the property details in the sidebar to get a price prediction.")

st.sidebar.header("Enter Property Features")
input_data = {}

if model: 
    with st.sidebar.form(key='prediction_form'):
        st.subheader("Numerical Features")
        
        for col in n_cols:
            input_data[col] = st.number_input(
                f"Enter {col.replace('_', ' ').title()}", 
                value=float(df_raw[col].mean()),
                format="%.2f"
            )
        
        for col in other_num_cols:
            input_data[col] = st.number_input(
                f"Enter Number of {col.title()}", 
                value=int(df_raw[col].median()), 
                min_value=int(df_raw[col].min()), 
                max_value=int(df_raw[col].max()), 
                step=1
            )
        
        st.subheader("Categorical Features")
        for col in ht_cols:
            input_data[col] = st.selectbox(
                f"Select {col.replace('_', ' ').title()}", 
                options=dropdown_options[col]
            )
        
        input_data['finishing_type'] = st.selectbox(
            "Select Finishing Type", 
            options=dropdown_options['finishing_type']
        )
        
        st.subheader("Amenities (Yes/No)")
        for col in l_cols:
            input_data[col] = st.selectbox(
                f"{col.replace('_', ' ').title()}?", 
                options=dropdown_options[col]
            )

        submit_button = st.form_submit_button(label='Predict Price')

    if submit_button:
        try:
            input_df = pd.DataFrame([input_data])
            input_processed = input_df.copy()


            mode_compound_count = df_raw['compound_count'].mode()[0]
            input_processed['compound_count'] = input_processed['compound_name'].map(compound_map).fillna(mode_compound_count)
            
            services_map = {'Yes': 1, 'No': 0}
            input_processed['services'] = (
                input_processed['has_security'].map(services_map) +
                input_processed['has_parking'].map(services_map) +
                input_processed['has_balcony'].map(services_map) +
                input_processed['has_amenities'].map(services_map)
            )

            finishing_map = {'Unfinished': 0, 'Semi-finished': 1, 'Lux': 2, 'Super Lux': 3}
            input_processed['finishing_type'] = input_processed['finishing_type'].map(finishing_map)

            for col in l_cols + ht_cols:
                le = encoders[col]
                input_val = input_processed[col].iloc[0]
                
                if input_val not in le.classes_:
                    fallback_value = df_raw[col].mode()[0]
                    st.warning(f"Value '{input_val}' for '{col}' is unknown (unseen during training). Using mode: '{fallback_value}'.")
                    input_processed[col] = fallback_value
                
                input_processed[col] = le.transform(input_processed[col])

            input_processed[n_cols] = scaler.transform(input_processed[n_cols])

            input_processed = input_processed[feature_cols]

            prediction = model.predict(input_processed)

            st.success(f"**Predicted Property Price: {prediction[0]:,.0f} EGP**")
        
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs and ensure all model files were generated correctly: {e}")

else:
    st.warning("Prediction service is unavailable. Please ensure `Model Training.py` has been run and all artifacts are deployed.")
