import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


FINISHING_MAP = {'Unfinished': 0, 'Semi-finished': 1, 'Lux': 2 , 'Super Lux': 3}

SCALED_COLS = ['area_sqm', 'floor_number', 'building_age_years', 'distance_to_auc_km',
               'distance_to_mall_km', 'distance_to_metro_km', 'days_on_market']

BINARY_MAP_YN = {'Yes': 1, "No": 0} 

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path)
    df = df.copy()

    df['compound_name'] = df['compound_name'].fillna(df['compound_name'].mode()[0])
    df.set_index('listing_id', inplace=True)
    df.drop('listing_date', axis=1, inplace=True)

    df['finishing_type'] = df['finishing_type'].map(FINISHING_MAP)
    
    df['services'] = (df['has_security'].map(BINARY_MAP_YN) + 
                      df['has_parking'].map(BINARY_MAP_YN) +
                      df['has_balcony'].map(BINARY_MAP_YN) + 
                      df['has_amenities'].map(BINARY_MAP_YN))
    
    lb_dict = {}
    
    lb_cols = ['has_balcony', 'has_parking', 'has_security', 'has_amenities','seller_type','is_negotiable',
               'district', 'compound_name','view_type']
    
    for col in lb_cols:
        lb = LabelEncoder()
        df[col] = lb.fit_transform(df[col])
        lb_dict[col] = lb
    
    return df.drop_duplicates(), lb_dict

def train_model(file_path):

    df, lb_dict = load_and_preprocess_data(file_path)

    X = df.drop('price_egp', axis=1)
    Y = df['price_egp']

    y_binned = pd.qcut(Y, q=5, labels=False, duplicates='drop')
    x_train, _, y_train, _ = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=y_binned
    )

    scaler = StandardScaler()
    x_train[SCALED_COLS] = scaler.fit_transform(x_train[SCALED_COLS])
    
    best_model = GradientBoostingRegressor(
        n_estimators=300, 
        max_depth=10, 
        learning_rate=0.05, 
        random_state=42
    )

    best_model.fit(x_train, y_train)

    return best_model, scaler, lb_dict
