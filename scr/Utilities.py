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

    columns_to_drop = ['listing_id', 'listing_date', 'days_on_market', 'seller_type', 'is_negotiable']
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
