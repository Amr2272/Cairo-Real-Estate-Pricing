import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


BINARY_MAP = {'Yes': 1, 'No': 0, 'Broker': 0, 'Owner': 1} 
FINISHING_MAP = {'Unfinished': 0, 'Semi-finished': 1, 'Lux': 2 , 'Super Lux': 3}

LABEL_ENCODED_COLS = ['district', 'compound_name', 'view_type', 'seller_type', 'is_negotiable']
SCALED_COLS = ['area_sqm', 'floor_number', 'building_age_years', 'distance_to_auc_km',
               'distance_to_mall_km', 'distance_to_metro_km', 'days_on_market']


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_and_split(df, test_size=0.1, random_state=42):
    df = df.copy()
    df['compound_name'] = df['compound_name'].fillna(df['compound_name'].mode()[0])
    df.set_index('listing_id', inplace=True)
    df.drop('listing_date', axis=1, inplace=True)

    df['finishing_type'] = df['finishing_type'].map(FINISHING_MAP)
    
    binary_cols = ['has_balcony', 'has_parking', 'has_security', 'has_amenities', 
                   'is_negotiable', 'seller_type']

    for col in binary_cols:
        df[col] = df[col].map(BINARY_MAP)

    df['services'] = (df['has_security'] + df['has_parking'] +
                      df['has_balcony'] + df['has_amenities'])

    lb_dict = {}
    for col in ['district', 'compound_name', 'view_type']:
        lb = LabelEncoder()
        df[col] = lb.fit_transform(df[col])
        lb_dict[col] = lb 

    X = df.drop('price_egp', axis=1)
    Y = df['price_egp']

    y_binned = pd.qcut(Y, q=5, labels=False, duplicates='drop')
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=y_binned
    )

    scaler = StandardScaler()
    x_train[SCALED_COLS] = scaler.fit_transform(x_train[SCALED_COLS])
    x_test[SCALED_COLS] = scaler.transform(x_test[SCALED_COLS])
    
    return x_train, x_test, y_train, y_test, scaler, lb_dict

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return r2, mae
