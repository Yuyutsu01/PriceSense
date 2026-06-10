"""
Orchestration Pipeline Script
=============================

Concept:
In ML engineering, orchestration is the process of coordinating and scheduling individual steps
(Data Loading -> EDA -> Data Cleaning & Preprocessing -> Model Training -> Evaluation & Tuning -> Explainability)
into a cohesive, reproducible workflow.

Architecture:
1. `src/data_loader.py` retrieves and caches the raw housing dataset.
2. `src/eda.py` analyzes data distributions and correlation trends, saving visual assets to disk.
3. `src/preprocessing.py` cleans the data (including removing massive outliers with GrLivArea > 4000)
   and engineers 20+ features, returning encoded train/validation datasets.
4. `src/train.py` performs 5-fold cross-validation on Linear Regression, Random Forest, and XGBoost,
   executes RandomizedSearchCV to find optimal hyperparameter spaces, evaluates on test data, and runs
   SHAP TreeExplainer for local/global explainability.
5. Save the trained pipeline assets (best XGBoost model + preprocessor metadata) to `models/`.
"""

import os
import joblib
import pandas as pd

from src.data_loader import get_raw_data
from src.eda import run_eda
from src.preprocessing import HousingPreprocessor
from src.train import train_and_evaluate

def main():
    print("=========================================================")
    print("   Starting Housing Price Prediction Machine Learning Pipeline ")
    print("=========================================================\n")
    
    # Step 1: Load/Fetch Ames Housing Data
    raw_df = get_raw_data(data_dir="data", filename="raw_housing.csv")
    print(f"Dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.\n")
    
    # Step 2: Exploratory Data Analysis
    # Generates price distribution, correlation heatmap, and scatter plots
    run_eda(raw_df, output_dir="static/plots")
    print()
    
    # Step 3: Instantiate Preprocessor & Run Cleaning/Feature Engineering
    preprocessor = HousingPreprocessor()
    
    # Remove extreme outliers (GrLivArea > 4000) to improve model learning and prevent high skew
    cleaned_df = preprocessor.clean_outliers(raw_df)
    
    # Impute missing values, engineer 22 domain-specific features, and one-hot encode categorical features
    print("Cleaning, engineering features, and encoding categories...")
    X_processed, y_processed = preprocessor.fit_transform(cleaned_df)
    print(f"Processed feature matrix shape: {X_processed.shape}\n")
    
    # Step 4: Model Training, CV, Tuning & Evaluation
    # Trains Linear Regression, Random Forest, XGBoost (Base), and Tuned XGBoost.
    # Outputs evaluation table, saves SHAP plots, and saves the best model.
    pipeline_results = train_and_evaluate(
        X=X_processed,
        y=y_processed,
        models_dir="models",
        plots_dir="static/plots"
    )
    
    # Step 5: Save fitted Preprocessor
    # We save the preprocessor to models/preprocessor.joblib so that app.py can load it
    # and perform identical imputation and one-hot encoding feature alignment.
    preprocessor_path = os.path.join("models", "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved fitted preprocessor to {preprocessor_path}\n")
    
    print("=========================================================")
    print("        Pipeline Execution Completed Successfully        ")
    print("=========================================================")

if __name__ == "__main__":
    main()
