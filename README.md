# Freight Rate Prediction Challenge

## Quick Setup

### 1. Install Python packages
```bash
pip install -r requirements.txt


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
 All data files renamed and  stored in the data/ folder:

    train_test.csv

    validation.csv

    validation_predictions_template.csv

    december_chart_inputs.csv

How to Run
Step 1: Run the prediction pipeline
```bash

python main.py

This will:

    Load and explore the data

    Create new features

    Train an XGBoost model

    Predict rates for 12,000 validation loads

    Create December predictions
	
	
	
	
📁 The following Files are Created based on the command ran above;
	Note: these files have already been created as step 1 and 2 have been executed before now,
however you may still run as it will override
✅ 1. validation_predictions.csv -  SUBMISSION FILE!

    12,000 predictions for the validation data

    Format: load_id, predicted_rate

    

✅ 2. candidate_december.png - THE DECEMBER CHART

    Shows predicted freight rates for December 2025

    

✅ 3. models/best_model.joblib -  TRAINED MODEL

    The XGBoost model that made all the predictions

    Can be used to make predictions on new data
	

✅ 4. models/preprocessor.joblib - PREPROCESSING PIPELINE

    Handles scaling, encoding, and imputation

    Makes sure new data is processed the same way
	

✅ 5. data/december_daily_averages.csv - DAILY AVERAGES

    Average predicted rate for each day in December

    Used to create the chart
	
	
✅ 5. data/december_chart_inputs_with_predictions.csv - DAILY AVERAGES

    Average predicted rate for each day in December

    Used to create the chart FOR scorer
	
	

Step 2: Run the scorer to validate my work
```bash

python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_with_predictions.csv

This will:

    Validate my submission file

    Validate my December predictions

    Create the official December chart in scorer_results/candidate_december.png



After running both steps it will generate:


    ✅ validation_predictions.csv which has 12,000 rows with correct IDs

    ✅ or validate  December predictions are correct

    ✅ Create scorer_results/candidate_december.png (official chart)

so the resulting files are : validation_predictions.csv  and  scorer_results/candidate_december.png
Check your outputs:

You should see:


Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.

Model Performance

    Model: XGBoost with hyperparameter tuning

    Features: 20+ engineered features

    Validation: 80/20 split with 5-fold cross-validation
	
	
	
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###





Quick Commands Summary


# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Validate your work and get official chart
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs_with_predictions.csv
