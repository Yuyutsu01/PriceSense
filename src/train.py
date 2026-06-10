"""
Model Training, Cross-Validation, Hyperparameter Tuning & Explainability Module
==============================================================================

Concept:
1. Model Comparison (Intermediate):
   - Linear Regression (Baseline): A simple model that assumes a linear relationship between features and target.
     It has high bias but low variance.
   - Random Forest (Intermediate): An ensemble bagging model that averages many decision trees to reduce variance.
     It handles non-linear relationships well out of the box.
   - XGBoost (Advanced): An extreme gradient boosting model that trains trees sequentially, where each tree corrects
     the errors of the previous ones. It is highly efficient and usually yields the best accuracy.

2. Cross-Validation (Intermediate):
   - K-Fold Cross Validation splits the training set into K folds. It trains the model K times, each time using K-1 folds
     for training and 1 fold for validation. This gives a robust estimate of performance and prevents overfitting.

3. Hyperparameter Tuning (Advanced):
   - We use RandomizedSearchCV to explore the hyperparameter space of XGBoost. It randomly samples combinations
     of hyperparameters, training and evaluating them using cross-validation. This is more efficient than GridSearchCV.

4. Explainability (Advanced):
   - SHAP (SHapley Additive exPlanations) is based on cooperative game theory. It attributes a payout (prediction change)
     to players (features) to explain the impact of each feature on the model's prediction.
   - Global Importance: The mean absolute SHAP value shows which features have the largest average impact.
   - Local Direction: A high feature value can push the price up (positive driver) or down (negative driver).

Architecture:
- The script trains all models, prints an evaluation comparison table, performs hyperparameter tuning,
  computes SHAP values, generates importance plots, and saves both the preprocessor and best model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import shap

def evaluate_predictions(y_true, y_pred):
    """
    Computes standard regression evaluation metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, models_dir: str = "models", plots_dir: str = "static/plots") -> dict:
    """
    Splits data, runs 5-fold CV, trains Linear Regression, Random Forest, and XGBoost,
    performs hyperparameter tuning for XGBoost, outputs a comparison table, calculates SHAP,
    and caches the best performing model.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Train-Test Split (80% Train, 20% Test for validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into Train: {X_train.shape} and Test: {X_test.shape}")
    
    # 2. Cross Validation Setup (5-Fold KFold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    # --- MODEL 1: Linear Regression (Baseline) ---
    print("\n--- Training Linear Regression (Baseline) ---")
    lr = LinearRegression()
    # Cross Validation
    cv_scores_lr = cross_val_score(lr, X_train, y_train, cv=kf, scoring='r2')
    print(f"Linear Regression 5-Fold CV R²: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")
    
    # Fit & Test
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr, mae_lr, r2_lr = evaluate_predictions(y_test, y_pred_lr)
    results['Linear Regression'] = {'RMSE': rmse_lr, 'MAE': mae_lr, 'R²': r2_lr}
    
    # --- MODEL 2: Random Forest (Intermediate) ---
    print("\n--- Training Random Forest (Intermediate) ---")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # Cross Validation
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='r2')
    print(f"Random Forest 5-Fold CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
    
    # Fit & Test
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf, mae_rf, r2_rf = evaluate_predictions(y_test, y_pred_rf)
    results['Random Forest'] = {'RMSE': rmse_rf, 'MAE': mae_rf, 'R²': r2_rf}
    
    # --- MODEL 3: XGBoost (Advanced) ---
    print("\n--- Training XGBoost (Advanced) ---")
    xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
    # Cross Validation
    cv_scores_xgb = cross_val_score(xgb_base, X_train, y_train, cv=kf, scoring='r2')
    print(f"XGBoost (Baseline) 5-Fold CV R²: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
    
    # --- MODEL 4: XGBoost Hyperparameter Tuning ---
    print("\n--- Tuning XGBoost Hyperparameters ---")
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb_tune = XGBRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=xgb_tune,
        param_distributions=param_dist,
        n_iter=15,
        scoring='r2',
        cv=kf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    print("Best XGBoost parameters found:")
    print(random_search.best_params_)
    
    best_xgb = random_search.best_estimator_
    
    # Cross Validation of Tuned Model
    cv_scores_xgb_tuned = cross_val_score(best_xgb, X_train, y_train, cv=kf, scoring='r2')
    print(f"XGBoost (Tuned) 5-Fold CV R²: {cv_scores_xgb_tuned.mean():.4f} (+/- {cv_scores_xgb_tuned.std():.4f})")
    
    # Fit & Test
    y_pred_xgb = best_xgb.predict(X_test)
    rmse_xgb, mae_xgb, r2_xgb = evaluate_predictions(y_test, y_pred_xgb)
    results['XGBoost (Tuned)'] = {'RMSE': rmse_xgb, 'MAE': mae_xgb, 'R²': r2_xgb}
    
    # 3. Model Comparison Table
    df_compare = pd.DataFrame(results).T
    print("\n=== Model Evaluation Comparison ===")
    print(df_compare.to_string())
    
    # Save comparison table as CSV and LaTeX/Markdown
    df_compare.to_csv(os.path.join(models_dir, "model_comparison.csv"))
    
    # 4. Explainability (SHAP)
    print("\n--- Generating SHAP Explainability ---")
    # Using TreeExplainer as it is highly optimized for tree ensembles like XGBoost
    explainer = shap.TreeExplainer(best_xgb)
    
    # Compute SHAP values for the test set
    print("Computing SHAP values on test set...")
    shap_values = explainer(X_test)
    
    # Plot summary and save it
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance Summary (XGBoost)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_plot_path = os.path.join(plots_dir, "shap_summary.png")
    plt.savefig(shap_plot_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary plot to {shap_plot_path}")
    
    # Identify positive and negative drivers from SHAP explanation object
    # We aggregate the mean absolute SHAP value for feature importance
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'MeanAbsSHAP': mean_shap
    }).sort_values(by='MeanAbsSHAP', ascending=False)
    
    # Let's save feature importance
    importance_df.to_csv(os.path.join(models_dir, "feature_importance_shap.csv"), index=False)
    print("\nTop 10 Most Important Features (by SHAP):")
    print(importance_df.head(10).to_string(index=False))
    
    # Save the best model
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_xgb, model_path)
    print(f"\nSaved best model (XGBoost) to {model_path}")
    
    return {
        'best_model': best_xgb,
        'comparison': df_compare,
        'importance': importance_df,
        'X_train_cols': X_train.columns.tolist()
    }
