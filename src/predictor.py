import pandas as pd
import numpy as np
import joblib

class Predictor:
    def __init__(self):
        self.model = None
    
    def load_model(self, model_path='models/best_model.joblib'):
        """Load the trained model"""
        self.model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not loaded! Please load the model first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def create_submission(self, template_df, predictions, output_path='validation_predictions.csv'):
        """Create the submission file"""
        submission = template_df.copy()
        submission['predicted_rate'] = predictions
        
        # Check if there were any NaN values
        if submission['predicted_rate'].isnull().sum() > 0:
            print(f"⚠️ Warning: {submission['predicted_rate'].isnull().sum()} NaN predictions found!")
            submission['predicted_rate'] = submission['predicted_rate'].fillna(submission['predicted_rate'].median())
        
        submission.to_csv(output_path, index=False)
        
        print(f"\n✅ Submission saved to {output_path}")
        print(f"   Shape: {submission.shape}")
        print(f"   Mean predicted rate: ${submission['predicted_rate'].mean():.2f}")
        print(f"   Min predicted rate: ${submission['predicted_rate'].min():.2f}")
        print(f"   Max predicted rate: ${submission['predicted_rate'].max():.2f}")
        
        return submission