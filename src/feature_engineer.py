import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.created_features = []
    
    def engineer_features(self, df):
        """Create new features from existing data"""
        df_new = df.copy()
        self.created_features = []
        
        # 1. Date features
        if 'date' in df_new.columns:
            try:
                dates = pd.to_datetime(df_new['date'])
                df_new['month'] = dates.dt.month
                df_new['day_of_week'] = dates.dt.dayofweek
                df_new['quarter'] = dates.dt.quarter
                df_new['is_weekend'] = dates.dt.dayofweek.isin([5, 6]).astype(int)
                df_new['day_of_year'] = dates.dt.dayofyear
                self.created_features.extend(['month', 'day_of_week', 'quarter', 'is_weekend', 'day_of_year'])
            except Exception as e:
                pass
        
        # 2. Distance features
        if 'distance' in df_new.columns:
            df_new['distance_squared'] = df_new['distance'] ** 2
            df_new['distance_log'] = np.log1p(df_new['distance'])
            df_new['distance_sqrt'] = np.sqrt(df_new['distance'])
            self.created_features.extend(['distance_squared', 'distance_log', 'distance_sqrt'])
        
        # 3. Weight features
        if 'weight' in df_new.columns:
            df_new['weight_squared'] = df_new['weight'] ** 2
            df_new['weight_log'] = np.log1p(df_new['weight'])
            df_new['weight_sqrt'] = np.sqrt(df_new['weight'])
            self.created_features.extend(['weight_squared', 'weight_log', 'weight_sqrt'])
        
        # 4. Distance-Weight interaction
        if 'distance' in df_new.columns and 'weight' in df_new.columns:
            df_new['distance_weight'] = df_new['distance'] * df_new['weight']
            df_new['distance_weight_ratio'] = df_new['distance'] / (df_new['weight'] + 1)
            self.created_features.extend(['distance_weight', 'distance_weight_ratio'])
        
        # 5. Location-based features
        if all(col in df_new.columns for col in ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']):
            df_new['lat_diff'] = df_new['delivery_lat'] - df_new['pickup_lat']
            df_new['lon_diff'] = df_new['delivery_lon'] - df_new['pickup_lon']
            self.created_features.extend(['lat_diff', 'lon_diff'])
        
        # 6. Distance per weight
        if 'distance' in df_new.columns and 'weight' in df_new.columns:
            df_new['distance_per_weight'] = df_new['distance'] / (df_new['weight'] + 1)
            self.created_features.append('distance_per_weight')
        
        return df_new
    
    def handle_outliers(self, df):
        """Fix extreme values"""
        df_clean = df.copy()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)
        
        return df_clean