"""
Streamlit Web Application
=========================

Concept:
Deployment is the final stage of the ML lifecycle. A machine learning model is useless unless stakeholders
can interact with it. We build a Streamlit web application that provides:
1. Interactive Inputs: Users can specify housing attributes.
2. Real-Time Prediction: Outputs the expected house price.
3. Uncertainty/Confidence Bounds: Estimates the range of prices using the model's test RMSE.
4. Local SHAP Explainability: Displays the features that drove *this specific house's* price up or down.
5. Historical Market EDA: Displays the exploratory plots generated from the training dataset.

Architecture:
- Caches model assets (XGBoost model and fitted Preprocessor) for high performance.
- Loads raw dataset dynamically to get valid values for selectors (e.g., Neighborhood names).
- Generates a dummy row based on median house values, modifies user-selected values, processes the row through
  the preprocessor, and invokes XGBoost to compute predictions and local SHAP values.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set page settings
st.set_page_config(
    page_title="PriceSense - Ames Housing Predictor",
    
    layout="wide"
)

# Custom premium styling using vanilla CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header design */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    
    /* Card design */
    .metric-card {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }
    
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4f46e5;
    }
    
    .metric-range {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 1. Cache loaders for high-performance predictions
@st.cache_resource
def load_ml_assets():
    """
    Loads and caches the best model and fitted preprocessor.
    """
    model_path = "models/best_model.joblib"
    preprocessor_path = "models/preprocessor.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        st.error("Error: Trained model assets not found! Please run the pipeline script `python run_pipeline.py` first.")
        st.stop()
        
    model = joblib.load(model_path)
    preproc = joblib.load(preprocessor_path)
    return model, preproc

@st.cache_data
def get_data_metadata():
    """
    Loads raw housing data to extract categorical fields (like Neighborhood lists)
    and evaluation metrics from the comparison table.
    """
    csv_path = "data/raw_housing.csv"
    compare_path = "models/model_comparison.csv"
    
    if not os.path.exists(csv_path):
        st.error("Error: Cached dataset raw_housing.csv not found! Run the pipeline first.")
        st.stop()
        
    df = pd.read_csv(csv_path)
    neighborhoods = sorted(df['Neighborhood'].dropna().unique().tolist())
    
    # Estimate standard error of prediction using test set RMSE
    rmse = 22000.0  # default fallback
    r2_score_val = 0.90
    if os.path.exists(compare_path):
        comp_df = pd.read_csv(compare_path)
        # Col 0 is usually name
        col0 = comp_df.columns[0]
        xgb_row = comp_df[comp_df[col0].str.contains("XGBoost", na=False)]
        if not xgb_row.empty:
            rmse = float(xgb_row['RMSE'].values[0])
            r2_score_val = float(xgb_row['R²'].values[0])
            
    return df, neighborhoods, rmse, r2_score_val

# Load assets
model, preprocessor = load_ml_assets()
raw_df, neighborhoods, model_rmse, model_r2 = get_data_metadata()

# Header banner
st.markdown("""
<div class="header-container">
    <h1 class="header-title">PriceSense 🏠</h1>
    <div class="header-subtitle">Advanced Ames Housing Valuation Engine powered by Tuned XGBoost & SHAP Explainability</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_predict, tab_market_eda = st.tabs(["🏠 House Valuation Engine", "📊 Historical Market EDA"])

with tab_predict:
    st.markdown("Enter the property attributes below to estimate the sale value and understand pricing drivers.")
    
    # Sidebar or Columns for Inputs
    st.sidebar.markdown('<div class="sidebar-header">Key Settings</div>', unsafe_allow_html=True)
    
    # Layout with three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📐 Size & Layout")
        gr_liv_area = st.number_input("Living Area (Above Ground sq ft)", min_value=300, max_value=6000, value=1500, step=50)
        tot_bsmt_sf = st.number_input("Basement Area (Total sq ft)", min_value=0, max_value=4000, value=1000, step=50)
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=480, step=20)
        lot_area = st.number_input("Lot Size (Total sq ft)", min_value=500, max_value=150000, value=10000, step=500)
        
    with col2:
        st.subheader("⭐ Quality & Construction")
        overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=6, help="Rates the overall material and finish of the house")
        overall_cond = st.slider("Overall Condition (1-10)", min_value=1, max_value=10, value=5, help="Rates the overall condition of the house")
        year_built = st.slider("Year Built", min_value=1872, max_value=2010, value=1995, step=1)
        year_remod = st.slider("Year Remodeled / Added", min_value=1950, max_value=2010, value=1995, step=1)
        
    with col3:
        st.subheader("📍 Location & Capacity")
        neighborhood = st.selectbox("Neighborhood Location", options=neighborhoods, index=neighborhoods.index("NAmes") if "NAmes" in neighborhoods else 0)
        bedrooms = st.slider("Bedrooms Above Grade", min_value=0, max_value=6, value=3)
        garage_cars = st.slider("Garage Car Capacity", min_value=0, max_value=4, value=2)
        
        # Bathrooms split
        st.markdown("**Bathrooms**")
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            full_bath = st.slider("Full Baths (Above Ground)", 0, 4, 2)
            bsmt_full_bath = st.slider("Full Baths (Basement)", 0, 3, 0)
        with b_col2:
            half_bath = st.slider("Half Baths (Above Ground)", 0, 2, 1)
            bsmt_half_bath = st.slider("Half Baths (Basement)", 0, 2, 0)

    # 2. Predict logic
    if st.button("Calculate Property Value", type="primary", use_container_width=True):
        # Construct raw DataFrame based on median profile
        # Use first row as a base template to populate columns not specified in our UI
        user_row = raw_df.iloc[0].copy()
        
        # Override with user inputs
        user_row['GrLivArea'] = gr_liv_area
        user_row['TotalBsmtSF'] = tot_bsmt_sf
        user_row['GarageArea'] = garage_area
        user_row['LotArea'] = lot_area
        user_row['OverallQual'] = overall_qual
        user_row['OverallCond'] = overall_cond
        user_row['YearBuilt'] = year_built
        user_row['YearRemodAdd'] = year_remod
        user_row['Neighborhood'] = neighborhood
        user_row['BedroomAbvGr'] = bedrooms
        user_row['GarageCars'] = garage_cars
        user_row['FullBath'] = full_bath
        user_row['HalfBath'] = half_bath
        user_row['BsmtFullBath'] = bsmt_full_bath
        user_row['BsmtHalfBath'] = bsmt_half_bath
        user_row['YrSold'] = 2010 # Dataset maximum
        
        # Drop SalePrice
        if 'SalePrice' in user_row.index:
            user_row = user_row.drop('SalePrice')
            
        user_df = pd.DataFrame([user_row])
        
        # Apply preprocessing
        X_user = preprocessor.transform(user_df)
        
        # Run prediction
        predicted_val = float(model.predict(X_user)[0])
        
        # Compute confidence interval (1.96 * RMSE of test set for a 95% prediction interval)
        lower_bound = max(0.0, predicted_val - 1.96 * model_rmse)
        upper_bound = predicted_val + 1.96 * model_rmse
        
        # Layout outputs
        st.markdown("---")
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estimated Market Value</h3>
            <div class="metric-val">${predicted_val:,.2f}</div>
            <div class="metric-range">95% Confidence Valuation Range: <b>${lower_bound:,.2f}</b> to <b>${upper_bound:,.2f}</b></div>
            <div class="metric-range" style="font-size:0.9rem; margin-top:0.5rem; opacity:0.8;">
                Model statistics: R² = {model_r2:.2%}, Standard Error = ${model_rmse:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # SHAP Explainability for the individual property
        st.subheader("💡 Valuation Explainer (Local SHAP Analysis)")
        st.write("This chart shows which attributes pushed this property's price higher (positive drivers) or lower (negative drivers) relative to the average house in the dataset.")
        
        try:
            # TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_user)
            
            # Map columns back to friendly labels
            def get_friendly_label(name):
                if name.startswith("Neighborhood_"):
                    return f"Neighborhood: {name.replace('Neighborhood_', '')}"
                # Clean up engineered names
                mapping = {
                    'GrLivArea': 'Above Ground Living Area',
                    'TotalBsmtSF': 'Basement Area',
                    'GarageArea': 'Garage Size (sq ft)',
                    'TotalArea': 'Total Combined Area',
                    'TotalBath': 'Total Bathrooms',
                    'HouseAge': 'Age of Property (years)',
                    'RemodelAge': 'Years since Remodel',
                    'QualScore': 'Combined Quality & Condition',
                    'OverallQual': 'Overall Quality Rating',
                    'OverallCond': 'Overall Condition Rating',
                    'LotArea': 'Lot Size (sq ft)',
                    'BedroomAbvGr': 'Bedrooms count',
                    'GarageCars': 'Garage Car capacity'
                }
                return mapping.get(name, name)
            
            # Extract and sort features by SHAP value
            shap_df = pd.DataFrame({
                'InternalFeature': X_user.columns,
                'SHAPValue': shap_values.values[0]
            })
            
            # Remove features that had zero impact
            shap_df = shap_df[shap_df['SHAPValue'] != 0].copy()
            shap_df['FeatureName'] = shap_df['InternalFeature'].apply(get_friendly_label)
            shap_df['AbsValue'] = shap_df['SHAPValue'].abs()
            shap_df = shap_df.sort_values(by='AbsValue', ascending=False).head(10)
            
            # Split positive vs negative
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ca02c' if val > 0 else '#d62728' for val in shap_df['SHAPValue']]
            
            # Plot
            sns.barplot(data=shap_df, x='SHAPValue', y='FeatureName', palette=colors, hue='FeatureName', legend=False, ax=ax)
            ax.axvline(x=0, color='#666666', linestyle='--', linewidth=0.8)
            ax.set_xlabel("Impact on Predicted Price ($)", fontsize=11, labelpad=10)
            ax.set_ylabel("")
            ax.set_title("Top 10 Valuation Drivers for this Property", fontsize=12, fontweight='bold', pad=10)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${int(x):,}"))
            
            # Highlight positive and negative bars
            plt.tight_layout()
            st.pyplot(fig)
            
            # Descriptive analysis of factors
            pos_drivers = shap_df[shap_df['SHAPValue'] > 0]
            neg_drivers = shap_df[shap_df['SHAPValue'] < 0]
            
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                st.markdown("🟢 **Top Positive Value Drivers**")
                if not pos_drivers.empty:
                    for idx, row in pos_drivers.head(3).iterrows():
                        st.markdown(f"- **{row['FeatureName']}**: Pushed price up by **+${row['SHAPValue']:,.2f}**")
                else:
                    st.markdown("*None*")
                    
            with e_col2:
                st.markdown("🔴 **Top Negative Value Drivers**")
                if not neg_drivers.empty:
                    for idx, row in neg_drivers.head(3).iterrows():
                        st.markdown(f"- **{row['FeatureName']}**: Pulled price down by **-${abs(row['SHAPValue']):,.2f}**")
                else:
                    st.markdown("*None*")
                    
        except Exception as e:
            st.warning("Could not calculate local SHAP explainability. Note: " + str(e))

with tab_market_eda:
    st.markdown("### Exploratory Data Analysis & Trends")
    st.write("Below are the findings and distributions from our analysis of the Ames Housing dataset.")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.subheader("💰 Price Distribution")
        st.write("Understanding the distribution of property prices. Notice the skewness indicates a few highly priced homes.")
        price_plot = "static/plots/price_distribution.png"
        if os.path.exists(price_plot):
            st.image(price_plot, caption="Distribution of Ames Housing Prices")
        else:
            st.write("Distribution plot missing. Run the pipeline first.")
            
    with col_e2:
        st.subheader("🔥 Correlation Matrix")
        st.write("Showing how key sizes and metrics correlate linearly with pricing. Grade Living Area is highly correlated.")
        corr_plot = "static/plots/correlation_heatmap.png"
        if os.path.exists(corr_plot):
            st.image(corr_plot, caption="Pearson Correlation Matrix")
        else:
            st.write("Heatmap plot missing. Run the pipeline first.")
            
    st.markdown("---")
    st.subheader("📐 House Attributes vs. Price")
    st.write("Analyzing how areas like lot size, living room size, garage capacity, and basement size affect the sale value.")
    scatter_plot = "static/plots/scatter_plots.png"
    if os.path.exists(scatter_plot):
        st.image(scatter_plot, caption="Scatter plots showing relationships with regression lines")
    else:
        st.write("Scatter plots missing. Run the pipeline first.")
