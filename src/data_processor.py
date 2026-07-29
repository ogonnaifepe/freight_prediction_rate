import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer_num = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.numeric_features = []
        self.categorical_features = []
        self.feature_columns = []
        
    def load_data(self, train_path, validation_path, template_path, december_path):
        """Load all CSV files"""
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        template_df = pd.read_csv(template_path)
        december_df = pd.read_csv(december_path)
        
        print(f"✅ Training: {len(train_df)} rows, {len(train_df.columns)} columns")
        print(f"✅ Validation: {len(validation_df)} rows")
        print(f"✅ Template: {len(template_df)} rows")
        print(f"✅ December: {len(december_df)} rows")
        
        return train_df, validation_df, template_df, december_df
    
    def identify_features(self, df, target_col='posted_rate'):
        """Find numeric and text columns"""
        # Numeric columns (numbers)
        self.numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target and ID columns from features
        exclude = ['load_id', target_col, 'predicted_rate']
        self.numeric_features = [col for col in self.numeric_features if col not in exclude]
        
        # Text columns
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_features = [col for col in self.categorical_features if col not in exclude]
        
        print(f"📊 Numeric features: {len(self.numeric_features)}")
        print(f"📝 Text features: {len(self.categorical_features)}")
        
        return self.numeric_features, self.categorical_features
    
    def preprocess_data(self, df, fit=True, target_col=None):
        """Clean and prepare data"""
        df_clean = df.copy()
        
        # IMPORTANT FIX: Only use numeric features that actually exist in this dataframe
        existing_numeric = [col for col in self.numeric_features if col in df_clean.columns]
        
        # Handle numeric columns that exist
        if existing_numeric:
            numeric_data = df_clean[existing_numeric].values
            
            if fit:
                # For training data - learn the imputation values
                numeric_data = self.imputer_num.fit_transform(numeric_data)
                numeric_data = self.scaler.fit_transform(numeric_data)
            else:
                # For new data - use learned values
                try:
                    numeric_data = self.imputer_num.transform(numeric_data)
                    numeric_data = self.scaler.transform(numeric_data)
                except:
                    # If transform fails, refit on this data
                    print("   ⚠️ Refitting imputer for this data...")
                    numeric_data = self.imputer_num.fit_transform(numeric_data)
                    numeric_data = self.scaler.fit_transform(numeric_data)
            
            df_clean[existing_numeric] = numeric_data
        
        # Handle text columns that exist
        existing_cat = [col for col in self.categorical_features if col in df_clean.columns]
        
        for col in existing_cat:
            # Fill missing with 'missing'
            df_clean[col] = df_clean[col].fillna('missing').astype(str)
            
            if fit:
                # For training - learn the encoding
                self.label_encoders[col] = LabelEncoder()
                df_clean[col] = self.label_encoders[col].fit_transform(df_clean[col])
            else:
                # For new data - use learned encoding
                encoder = self.label_encoders[col]
                known_classes = set(encoder.classes_)
                
                def encode_value(x):
                    if x in known_classes:
                        return encoder.transform([x])[0]
                    else:
                        return -1  # Unseen category
                
                df_clean[col] = df_clean[col].apply(encode_value)
        
        # Build feature columns list
        self.feature_columns = existing_numeric + existing_cat
        
        # Separate target if it exists
        if target_col and target_col in df_clean.columns:
            y = df[target_col].values
            X = df_clean[self.feature_columns] if self.feature_columns else pd.DataFrame()
            return X, y
        else:
            # For prediction data - return what we have
            if self.feature_columns:
                X = df_clean[self.feature_columns]
            else:
                X = pd.DataFrame()
            return X, None
    
    def align_features(self, X, feature_columns):
        """Make sure all data has the same features"""
        # Create a new DataFrame with all required columns
        X_aligned = pd.DataFrame(index=X.index)
        
        for col in feature_columns:
            if col in X.columns:
                X_aligned[col] = X[col]
            else:
                # Add missing column with 0
                X_aligned[col] = 0
                print(f"   ➕ Added missing column: {col} (filled with 0)")
        
        return X_aligned
    
    def split_data(self, X, y, test_size=0.2):
        """Split into train and validation"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"📊 Training: {len(X_train)} samples")
        print(f"📊 Validation: {len(X_val)} samples")
        return X_train, X_val, y_train, y_val
    
    def save_preprocessor(self, path='models/preprocessor.joblib'):
        """Save preprocessor"""
        preprocessor_data = {
            'scaler': self.scaler,
            'imputer': self.imputer_num,
            'label_encoders': self.label_encoders,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_columns': self.feature_columns
        }
        joblib.dump(preprocessor_data, path)
        print(f"✅ Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='models/preprocessor.joblib'):
        """Load preprocessor"""
        preprocessor_data = joblib.load(path)
        self.scaler = preprocessor_data['scaler']
        self.imputer_num = preprocessor_data['imputer']
        self.label_encoders = preprocessor_data['label_encoders']
        self.numeric_features = preprocessor_data['numeric_features']
        self.categorical_features = preprocessor_data['categorical_features']
        self.feature_columns = preprocessor_data['feature_columns']
        print(f"✅ Preprocessor loaded from {path}")