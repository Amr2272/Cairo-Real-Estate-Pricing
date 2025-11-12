import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath="cairo_real_estate_dataset.csv"):

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return None


    df['compound_name'] = df['compound_name'].fillna('Not Specified')
    df['finishing_type'] = df['finishing_type'].fillna('Unfinished')

    df['listing_date'] = pd.to_datetime(df['listing_date'], errors='coerce')


    df['compound_count'] = df.groupby('compound_name')['listing_id'].transform('count')


    df['services'] = df['has_security'].map({'Yes':1,"No":0}).fillna(0) + \
                   df['has_parking'].map({'Yes':1,"No":0}).fillna(0) + \
                   df['has_balcony'].map({'Yes':1,"No":0}).fillna(0) + \
                   df['has_amenities'].map({'Yes':1,"No":0}).fillna(0)

    columns_to_drop = [
        'listing_id',
        'listing_date',
        'days_on_market',
        'seller_type',
        'is_negotiable',
        'has_balcony',
        'has_parking',
        'has_security',
        'has_amenities'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    df = df.dropna()

    return df

def get_preprocessor(categorical_features, numeric_features):

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor
