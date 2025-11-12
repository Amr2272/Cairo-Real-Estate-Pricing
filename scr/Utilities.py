import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath='cairo_real_estate_dataset.csv'):
    df = pd.read_csv(filepath)
    df['compound_name'] = df['compound_name'].fillna(df['compound_name'].mode()[0])
    return df

def get_column_lists():
    l_cols = [
        'has_balcony', 'has_parking', 'has_security', 
        'has_amenities', 'seller_type', 'is_negotiable'
    ]

    ht_cols = ['district', 'compound_name', 'view_type']
    
    n_cols = [
        'area_sqm', 'floor_number', 'building_age_years', 
        'distance_to_auc_km', 'distance_to_mall_km', 
        'distance_to_metro_km', 'days_on_market'
    ]
    
    other_num_cols = ['bedrooms', 'bathrooms']
    
    finishing_types = ['Unfinished', 'Semi-finished', 'Lux', 'Super Lux']
    
    return l_cols, ht_cols, n_cols, other_num_cols, finishing_types

def preprocess_for_training(df):

    compound_count_map = df.groupby('compound_name')['listing_id'].count().to_dict()
    df['compound_count'] = df.groupby('compound_name')['listing_id'].transform('count')
    
    services_map = {'Yes': 1, "No": 0}
    df['services'] = (df['has_security'].map(services_map) + 
                      df['has_parking'].map(services_map) + 
                      df['has_balcony'].map(services_map) + 
                      df['has_amenities'].map(services_map))
    
    df.set_index('listing_id', inplace=True)
    if 'listing_date' in df.columns:
        df.drop('listing_date', axis=1, inplace=True)

    df['finishing_type'] = df['finishing_type'].map({'Unfinished': 0, 'Semi-finished': 1, 'Lux': 2, 'Super Lux': 3})

    l_cols, ht_cols, n_cols, _, _ = get_column_lists()

    label_encoders = {}
    for col in l_cols + ht_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    X = df.drop('price_egp', axis=1)
    Y = df['price_egp']

    y_binned = pd.qcut(Y, q=5, labels=False, duplicates='drop')
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=y_binned
    )

    scaler = StandardScaler()
    x_train[n_cols] = scaler.fit_transform(x_train[n_cols])
    x_test[n_cols] = scaler.transform(x_test[n_cols])
    
    return x_train, x_test, y_train, y_test, scaler, label_encoders, compound_count_map

def get_feature_order(df):

    compound_count_map = df.groupby('compound_name')['listing_id'].count().to_dict()
    df['compound_count'] = df.groupby('compound_name')['listing_id'].transform('count')
    
    services_map = {'Yes': 1, "No": 0}
    df['services'] = (df['has_security'].map(services_map) + 
                      df['has_parking'].map(services_map) + 
                      df['has_balcony'].map(services_map) + 
                      df['has_amenities'].map(services_map))
    
    df.set_index('listing_id', inplace=True)
    if 'listing_date' in df.columns:
        df.drop('listing_date', axis=1, inplace=True)

    df['finishing_type'] = df['finishing_type'].map({'Unfinished': 0, 'Semi-finished': 1, 'Lux': 2, 'Super Lux': 3})

    l_cols, ht_cols, _, _, _ = get_column_lists()

    for col in l_cols + ht_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('price_egp', axis=1)
    return list(X.columns)