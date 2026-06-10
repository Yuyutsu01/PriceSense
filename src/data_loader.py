"""
Data Loader Module for Ames Housing Dataset
===========================================

Concept:
Caching is an intermediate software engineering concept where we store the results of expensive operations
(like downloading 80 features for 1460 houses over the internet) locally, so subsequent runs are fast and reliable.

Architecture:
1. Check if the local cache directory `data/` exists. If not, create it.
2. Check if `data/raw_housing.csv` exists.
   - Yes: Load it directly using pandas (O(1) network cost).
   - No: Use sklearn's `fetch_openml` to download dataset 42165 (Ames Housing), combine the features
     and the target ('SalePrice') into a single DataFrame, and write it to disk.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml

def get_raw_data(data_dir: str = "data", filename: str = "raw_housing.csv") -> pd.DataFrame:
    """
    Retrieves the Ames Housing dataset. Downloads from OpenML if not cached locally,
    otherwise loads the cached CSV.
    
    Args:
        data_dir (str): Directory where the raw dataset should be stored.
        filename (str): Name of the CSV file.
        
    Returns:
        pd.DataFrame: Merged dataset containing both features and target variable.
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, filename)
    
    if os.path.exists(cache_path):
        print(f"Loading dataset from local cache: {cache_path}")
        return pd.read_csv(cache_path)
    
    print("Fetching Ames Housing dataset (ID: 42165) from OpenML. This might take a minute...")
    # Fetching with as_frame=True to obtain pandas objects
    housing = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    
    # Merge features and target into one single dataframe
    df = housing.data.copy()
    
    # Add target 'SalePrice' (usually called 'SalePrice' or returned as housing.target)
    target_name = housing.target.name if housing.target.name else "SalePrice"
    df[target_name] = housing.target
    
    # Save to CSV cache
    df.to_csv(cache_path, index=False)
    print(f"Dataset cached successfully at {cache_path}")
    
    return df

if __name__ == "__main__":
    # Test execution
    data = get_raw_data()
    print(f"Data shape: {data.shape}")
