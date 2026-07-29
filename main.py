import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import Predictor

def create_december_chart(december_df, predictions):
    """Create the December 2026 prediction chart"""
    try:
        # Make a copy and add predictions
        chart_df = december_df.copy()
        chart_df['predicted_rate'] = predictions
        
        # Convert date
        if 'date' in chart_df.columns:
            chart_df['date'] = pd.to_datetime(chart_df['date'])
            
            # Group by date (should be daily already)
            daily_avg = chart_df.groupby(chart_df['date'].dt.date)['predicted_rate'].mean()
            
            # Create the chart
            plt.figure(figsize=(14, 7))
            plt.plot(daily_avg.index, daily_avg.values, 
                    marker='o', linewidth=2.5, markersize=8, color='darkblue')
            plt.title('Average Predicted Freight Rates - December 2025', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Average Predicted Rate ($)', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.savefig('candidate_december.png', dpi=300)
            plt.close()
            print("✅ December chart saved as 'candidate_december.png'")
            
            # Save daily averages
            daily_avg.to_csv('data/december_daily_averages.csv')
            print("✅ Daily averages saved to 'data/december_daily_averages.csv'")
        else:
            print("⚠️ No 'date' column found for December chart")
            
    except Exception as e:
        print(f"❌ Error creating December chart: {e}")

def main():
    print("="*70)
    print("🚚 FREIGHT RATE PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize components
    processor = DataProcessor()
    engineer = FeatureEngineer()
    trainer = ModelTrainer(random_state=42)
    predictor = Predictor()
    
    # 1. LOAD DATA
    print("\n📂 STEP 1: Loading data...")
    train_df, validation_df, template_df, december_df = processor.load_data(
        'data/train_test.csv',
        'data/validation.csv',
        'data/validation_predictions_template.csv',
        'data/december_chart_inputs.csv'
    )
    
    # 2. FEATURE ENGINEERING
    print("\n🔧 STEP 2: Feature engineering...")
    print("\nTraining data:")
    train_engineered = engineer.engineer_features(train_df)
    train_engineered = engineer.handle_outliers(train_engineered)
    
    print("\nValidation data:")
    val_engineered = engineer.engineer_features(validation_df)
    val_engineered = engineer.handle_outliers(val_engineered)
    
    print("\nDecember data:")
    dec_engineered = engineer.engineer_features(december_df)
    dec_engineered = engineer.handle_outliers(dec_engineered)
    
    # 3. DATA PREPROCESSING
    print("\n🧹 STEP 3: Preprocessing data...")
    
    # Identify features from training data
    numeric_cols, cat_cols = processor.identify_features(train_engineered, target_col='posted_rate')
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(cat_cols)}")
    
    # Preprocess training data (fit mode)
    print("\n   Preprocessing training data...")
    X_train_raw, y_train = processor.preprocess_data(
        train_engineered, fit=True, target_col='posted_rate'
    )
    
    # Preprocess validation data (transform mode)
    print("\n   Preprocessing validation data...")
    X_val_raw, _ = processor.preprocess_data(
        val_engineered, fit=False, target_col=None
    )
    
    # Preprocess December data (transform mode)
    print("\n   Preprocessing December data...")
    X_dec_raw, _ = processor.preprocess_data(
        dec_engineered, fit=False, target_col=None
    )
    
    # Get the feature columns from training
    feature_cols = processor.feature_columns
    print(f"\n   Total features: {len(feature_cols)}")
    
    # Align all datasets to have the same features
    print("\n   Aligning features for all datasets...")
    X_train = processor.align_features(X_train_raw, feature_cols)
    X_val = processor.align_features(X_val_raw, feature_cols)
    X_dec = processor.align_features(X_dec_raw, feature_cols)
    
    # Save preprocessor
    processor.save_preprocessor('models/preprocessor.joblib')
    
    # 4. SPLIT DATA
    print("\n✂️ STEP 4: Splitting data...")
    X_train_split, X_val_split, y_train_split, y_val_split = processor.split_data(
        X_train, y_train, test_size=0.2
    )
    
    # 5. MODEL TRAINING
    print("\n🤖 STEP 5: Training models...")
    
    # Train multiple models
    best_model = trainer.train_and_evaluate(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    
    # Hyperparameter tuning
    best_model = trainer.hyperparameter_tuning(X_train_split, y_train_split)
    
    # Cross-validation
    trainer.cross_validate(X_train_split, y_train_split)
    
    # Feature importance
    trainer.feature_importance(X_train_split, X_train_split.columns.tolist())
    
    # Save the best model
    trainer.save_model('models/best_model.joblib')
    
    # 6. MAKE PREDICTIONS
    print("\n🔮 STEP 6: Making predictions...")
    
    # Use the trained model
    predictor.model = best_model
    
    # Predict validation data
    print("   Predicting validation data...")
    val_predictions = predictor.predict(X_val)
    print(f"   ✅ Validation predictions: {len(val_predictions)}")
    
    # 7. CREATE SUBMISSION FILE
    print("\n📝 STEP 7: Creating submission file...")
    submission = predictor.create_submission(
        template_df, val_predictions, 'validation_predictions.csv'
    )
    
    # 8. DECEMBER PREDICTIONS
    print("\n📅 STEP 8: Predicting December rates...")
    dec_predictions = predictor.predict(X_dec)
    print(f"   ✅ December predictions: {len(dec_predictions)}")
    
    # SAVE DECEMBER PREDICTIONS FOR SCORER
    print("\n   Saving December predictions for scorer...")
    december_with_preds = december_df.copy()
    december_with_preds['predicted_rate'] = dec_predictions
    # Keep only the columns score.py expects
    columns_needed = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
    december_with_preds[columns_needed].to_csv(
        'data/december_chart_inputs_with_predictions.csv', index=False
    )
    print("   ✅ Saved to: data/december_chart_inputs_with_predictions.csv")
    
    # 9. CREATE DECEMBER CHART
    print("\n📊 STEP 9: Creating December chart...")
    create_december_chart(december_df, dec_predictions)
    
    # 10. FINAL SUMMARY
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Files created:")
    print("   1. validation_predictions.csv - Your submission file (12,000 predictions)")
    print("   2. candidate_december.png - December 2025 prediction chart")
    print("   3. models/best_model.joblib - Your trained model")
    print("   4. models/preprocessor.joblib - Preprocessing pipeline")
    print("   5. data/december_daily_averages.csv - Daily averages for December")
    print("   6. data/december_chart_inputs_with_predictions.csv - For scorer validation")
    print("\n✅ All done! Now run the scorer to validate your work.")
    print("   Command: python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_with_predictions.csv")

if __name__ == "__main__":
    main()