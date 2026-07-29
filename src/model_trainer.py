import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.results = pd.DataFrame()
        self.models = {}
    
    def define_models(self):
        """Define models to test"""
        self.models = {
            'Ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.01, random_state=self.random_state),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        return self.models
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val):
        """Train all models and find the best one"""
        self.define_models()
        
        results_list = []
        
        print("\n" + "="*60)
        print("?? TRAINING MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n?? Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            val_mae = mean_absolute_error(y_val, y_pred_val)
            val_r2 = r2_score(y_val, y_pred_val)
            
            results_list.append({
                'Model': name,
                'Train_RMSE': train_rmse,
                'Val_RMSE': val_rmse,
                'Val_MAE': val_mae,
                'Val_R2': val_r2,
                'Model_Object': model
            })
            
            print(f"   Train RMSE: {train_rmse:.2f}")
            print(f"   Val RMSE: {val_rmse:.2f}")
            print(f"   Val R: {val_r2:.4f}")
        
        # Find best model (lowest validation RMSE)
        self.results = pd.DataFrame(results_list)
        best_idx = self.results['Val_RMSE'].idxmin()
        self.best_model = self.results.loc[best_idx, 'Model_Object']
        best_name = self.results.loc[best_idx, 'Model']
        
        print("\n" + "="*60)
        print(f"?? BEST MODEL: {best_name}")
        print(f"   Validation RMSE: {self.results.loc[best_idx, 'Val_RMSE']:.2f}")
        print(f"   Validation R: {self.results.loc[best_idx, 'Val_R2']:.4f}")
        print("="*60)
        
        return self.best_model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Find the best settings for XGBoost"""
        print("\n" + "="*60)
        print("?? HYPERPARAMETER TUNING")
        print("="*60)
        
        model = XGBRegressor(random_state=self.random_state, n_jobs=-1)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        
        print(f"\n? Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        best_score = np.sqrt(-grid_search.best_score_)
        print(f"\n?? Best CV RMSE: {best_score:.2f}")
        
        return self.best_model
    
    def cross_validate(self, X_train, y_train, cv=5):
        """Cross-validation"""
        print("\n" + "="*60)
        print("?? CROSS-VALIDATION")
        print("="*60)
        
        cv_scores = cross_val_score(
            self.best_model, X_train, y_train, 
            cv=cv, scoring='neg_mean_squared_error'
        )
        rmse_scores = np.sqrt(-cv_scores)
        
        print(f"\n?? {cv}-fold CV RMSE:")
        print(f"   Mean: {rmse_scores.mean():.2f}")
        print(f"   Std: {rmse_scores.std():.2f}")
        print(f"   Min: {rmse_scores.min():.2f}")
        print(f"   Max: {rmse_scores.max():.2f}")
        
        return rmse_scores
    
    def feature_importance(self, X_train, feature_names):
        """Show which features are most important"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\n" + "="*60)
            print("?? TOP 10 MOST IMPORTANT FEATURES")
            print("="*60)
            for i, row in importance_df.head(10).iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.4f}")
            
            return importance_df
        return None
    
    def save_model(self, path='models/best_model.joblib'):
        """Save the model"""
        joblib.dump(self.best_model, path)
        print(f"? Model saved to {path}")
    
    def load_model(self, path='models/best_model.joblib'):
        """Load a saved model"""
        self.best_model = joblib.load(path)
        print(f"? Model loaded from {path}")
        return self.best_model