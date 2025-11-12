import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Utilities import load_data, get_column_lists

def run_eda():

    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("---")

    @st.cache_data
    def get_data_for_eda():
        df = load_data('cairo_real_estate_dataset.csv')
        

        df['compound_count'] = df.groupby('compound_name')['listing_id'].transform('count')
        
        services_map = {'Yes': 1, "No": 0}
        df['services'] = (
            df['has_security'].map(services_map) +
            df['has_parking'].map(services_map) +
            df['has_balcony'].map(services_map) +
            df['has_amenities'].map(services_map)
        )
        
        df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')

        return df

    df = get_data_for_eda()
    _, _, n_cols, other_num_cols, _ = get_column_lists()
    numerical_cols = n_cols + other_num_cols


    st.header("1. Data Overview")

    if st.checkbox("Show Raw Data and Shape"):
        st.subheader("Raw Data Sample")
        st.dataframe(df.head())
        st.write(f"Data Shape: **{df.shape[0]} rows**, **{df.shape[1]} columns**")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values Check")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if missing_values.empty:
            st.success("No missing values found after initial cleaning.")
        else:
            st.dataframe(missing_values)

    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T)

   
    st.header("2. Key Visualizations")

    st.subheader("Distribution of Price (EGP)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['price_egp'], kde=True, ax=ax)
    ax.set_title('Distribution of Price (EGP)')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Price vs. Area (sqm)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x='area_sqm', y='price_egp', data=df, ax=ax)
    ax.set_title('Price vs. Area')
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Price Distribution by Key Categorical Features")
    
    cat_cols = ['district', 'finishing_type', 'seller_type', 'view_type']
    
    for col in cat_cols:
        st.markdown(f"**{col.replace('_', ' ').title()}**")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=col, y='price_egp', data=df, ax=ax)
        plt.xticks(rotation=45, ha='right')
        ax.set_title(f'Price Distribution by {col.replace("_", " ").title()}')
        st.pyplot(fig)
        plt.close(fig)

if __name__ == '__main__':
    run_eda()