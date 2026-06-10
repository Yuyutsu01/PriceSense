"""
Data Preprocessing & Feature Engineering Module
===============================================

Concept:
1. Data Cleaning (Beginner):
   - Missing values are filled. Categorical columns with missing values (like PoolQC or GarageType) often mean the
     house does not have that feature, so we fill them with "None". For numerical columns (like LotFrontage), we
     impute them with the median to avoid skewing the distribution.
   - Outliers are identified and removed. We remove houses with GrLivArea > 4000 sq ft, which are recommended to be
     removed because they represent extreme, non-representative observations (massive houses sold at low prices).

2. Feature Engineering (Intermediate):
   - We construct domain-specific indicators that combine features to create more predictive predictors.
   - House Age and Remodeling Age tell us how old the property is and how recently it was updated.
   - Total Bathrooms aggregates full and half baths (both above ground and in basement).
   - Total Area combines living area, basement area, and garage area to capture the complete size of the home.
   - Quality Score interacts the overall quality and condition of the house.
   - We engineer 20+ features to maximize XGBoost and Random Forest predictive performance.

3. Categorical Encoding (Advanced):
   - Machine learning algorithms require numeric inputs. We use One-Hot Encoding to convert categorical features
     into multiple binary columns.
   - To deploy the model, the preprocessing steps must be reproducible. We build a `HousingPreprocessor` class
     that saves the imputation states and dummy column layouts from the training set, allowing us to preprocess
     arbitrary new data points (e.g., in Streamlit) identically.

Architecture:
- The `HousingPreprocessor` class exposes `fit_transform` and `transform` APIs.
- State variables (medians, modes, expected dummy columns) are stored as instance attributes.
"""

import pandas as pd
import numpy as np

class HousingPreprocessor:
    """
    A robust class that cleans data, performs feature engineering, and encodes categorical columns.
    Ensures consistency between training and inference data.
    """
    def __init__(self):
        self.medians = {}
        self.modes = {}
        self.training_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.target_name = "SalePrice"

    def clean_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes extreme outliers (e.g. GrLivArea > 4000) during training.
        """
        if 'GrLivArea' in df.columns:
            before_len = len(df)
            # Dean De Cock (dataset creator) recommends removing houses with GrLivArea > 4000
            df = df[df['GrLivArea'] <= 4000].copy()
            after_len = len(df)
            print(f"Outlier removal (GrLivArea > 4000): Removed {before_len - after_len} rows.")
        return df

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fits the preprocessor on training data, cleans it, engineers features, and encodes categories.
        Returns preprocessed features X and target y.
        """
        # 1. Separate target
        if self.target_name in df.columns:
            y = df[self.target_name].copy()
            X = df.drop(columns=[self.target_name]).copy()
        else:
            y = None
            X = df.copy()

        # Identify column types
        self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # 2. Impute missing values
        # Specific imputation rules:
        # LotFrontage -> Median
        if 'LotFrontage' in X.columns:
            self.medians['LotFrontage'] = X['LotFrontage'].median()
            # Advanced: fill LotFrontage by Neighborhood median if possible
            if 'Neighborhood' in X.columns:
                self.neighborhood_frontage_medians = X.groupby('Neighborhood')['LotFrontage'].median()
            else:
                self.neighborhood_frontage_medians = None

        # GarageType -> 'None', PoolQC -> 'None'
        # Let's save standard modes and medians for all other columns
        for col in self.numerical_cols:
            if col != 'LotFrontage':
                self.medians[col] = X[col].median()
                
        for col in self.categorical_cols:
            self.modes[col] = X[col].mode().iloc[0] if not X[col].mode().empty else "None"

        # Apply imputation
        X_clean = self._impute_data(X)

        # 3. Feature Engineering (20+ features)
        X_engineered = self._engineer_features(X_clean)

        # 4. One-Hot Encoding
        # Convert all categorical columns into dummies
        X_encoded = pd.get_dummies(X_engineered, columns=self.categorical_cols, drop_first=False)
        
        # Save training columns list for alignment during predict/transform
        self.training_cols = X_encoded.columns.tolist()

        return X_encoded, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new data (inference/test set) using fitted parameters.
        """
        # Ensure we don't accidentally modify the input
        X = df.copy()
        if self.target_name in X.columns:
            X = X.drop(columns=[self.target_name])

        # 1. Impute
        X_clean = self._impute_data(X)

        # 2. Engineer features
        X_engineered = self._engineer_features(X_clean)

        # 3. One-Hot Encoding
        X_encoded = pd.get_dummies(X_engineered, columns=self.categorical_cols, drop_first=False)

        # 4. Align columns with training set (adds missing dummies with 0, drops extra ones)
        missing_cols = [col for col in self.training_cols if col not in X_encoded.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=X_encoded.index, columns=missing_cols)
            X_encoded = pd.concat([X_encoded, missing_df], axis=1)
                
        # Drop columns not present in training
        X_encoded = X_encoded[self.training_cols].copy()

        return X_encoded

    def _impute_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs missing value imputation based on rules and fitted medians/modes.
        """
        df_imputed = df.copy()
        
        # Specific imputation: LotFrontage (Neighborhood-based or overall median)
        if 'LotFrontage' in df_imputed.columns:
            if hasattr(self, 'neighborhood_frontage_medians') and self.neighborhood_frontage_medians is not None:
                # Use neighborhood median
                df_imputed['LotFrontage'] = df_imputed.apply(
                    lambda row: self.neighborhood_frontage_medians.get(row['Neighborhood'], self.medians['LotFrontage']) 
                    if pd.isna(row['LotFrontage']) else row['LotFrontage'], 
                    axis=1
                )
            else:
                df_imputed['LotFrontage'] = df_imputed['LotFrontage'].fillna(self.medians['LotFrontage'])

        # Specific categorical columns commonly meaning "None"
        special_cat_cols = ['GarageType', 'PoolQC', 'Alley', 'BsmtQual', 'BsmtCond', 
                            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 
                            'GarageFinish', 'GarageQual', 'GarageCond', 'Fence', 'MiscFeature']
                            
        for col in special_cat_cols:
            if col in df_imputed.columns:
                # pandas handles categories carefully, check if it's categorical dtype
                if isinstance(df_imputed[col].dtype, pd.CategoricalDtype):
                    if "None" not in df_imputed[col].cat.categories:
                        df_imputed[col] = df_imputed[col].cat.add_categories("None")
                df_imputed[col] = df_imputed[col].fillna("None")

        # General imputation for numerical columns
        for col in self.numerical_cols:
            if col in df_imputed.columns:
                # For basement/garage/masonry areas, missing values typically mean 0 (no basement/garage/masonry)
                if any(x in col.lower() for x in ['bsmt', 'garage', 'masvnr', 'pool']):
                    df_imputed[col] = df_imputed[col].fillna(0)
                else:
                    df_imputed[col] = df_imputed[col].fillna(self.medians.get(col, 0))

        # General imputation for categorical columns
        for col in self.categorical_cols:
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(self.modes.get(col, "None"))

        return df_imputed

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates 20+ domain-specific housing features.
        """
        df_eng = df.copy()

        # Let's define helper variables to make engineering safer (checking column existence)
        def get_col(col_name, default=0):
            return df_eng[col_name] if col_name in df_eng.columns else default

        # 1. House Age (YrSold - YearBuilt)
        df_eng['HouseAge'] = get_col('YrSold', 2010) - get_col('YearBuilt', 2000)
        # Prevent negative age due to data recording errors
        df_eng['HouseAge'] = df_eng['HouseAge'].clip(lower=0)

        # 2. Remodeling Age (YrSold - YearRemodAdd)
        df_eng['RemodelAge'] = get_col('YrSold', 2010) - get_col('YearRemodAdd', 2000)
        df_eng['RemodelAge'] = df_eng['RemodelAge'].clip(lower=0)

        # 3. Total Bathrooms (Full + 0.5*Half + BsmtFull + 0.5*BsmtHalf)
        df_eng['TotalBath'] = (
            get_col('FullBath') + 
            0.5 * get_col('HalfBath') + 
            get_col('BsmtFullBath') + 
            0.5 * get_col('BsmtHalfBath')
        )

        # 4. Total Area (GrLivArea + TotalBsmtSF + GarageArea)
        df_eng['TotalArea'] = get_col('GrLivArea') + get_col('TotalBsmtSF') + get_col('GarageArea')

        # 5. Quality Score (OverallQual * OverallCond)
        df_eng['QualScore'] = get_col('OverallQual', 5) * get_col('OverallCond', 5)

        # Let's add 15+ more features to satisfy "Engineered 20+ domain-specific features"
        # 6. IsNew (Property sold in the same year it was built)
        df_eng['IsNew'] = (get_col('YrSold') == get_col('YearBuilt')).astype(int)

        # 7. IsRemodeled (Remodel date is different from build date)
        df_eng['IsRemodeled'] = (get_col('YearRemodAdd') != get_col('YearBuilt')).astype(int)

        # 8. HasPool (Does the house have a pool area?)
        df_eng['HasPool'] = (get_col('PoolArea') > 0).astype(int)

        # 9. HasGarage (Does the house have a garage?)
        df_eng['HasGarage'] = (get_col('GarageArea') > 0).astype(int)

        # 10. HasBasement (Does the house have a basement?)
        df_eng['HasBasement'] = (get_col('TotalBsmtSF') > 0).astype(int)

        # 11. HasFireplace (Does the house have a fireplace?)
        df_eng['HasFireplace'] = (get_col('Fireplaces') > 0).astype(int)

        # 12. TotalPorchSF (Sum of wood deck and various porches)
        df_eng['TotalPorchSF'] = (
            get_col('WoodDeckSF') + 
            get_col('OpenPorchSF') + 
            get_col('EnclosedPorch') + 
            get_col('3SsnPorch') + 
            get_col('ScreenPorch')
        )

        # 13. HasPorch (Binary indicator for any porch)
        df_eng['HasPorch'] = (df_eng['TotalPorchSF'] > 0).astype(int)

        # 14. SqFtPerRoom (Average size of above-grade rooms)
        tot_rooms = get_col('TotRmsAbvGrd')
        # Use tot_rooms but clip lower at 1 to prevent division by zero
        df_eng['SqFtPerRoom'] = get_col('GrLivArea') / tot_rooms.clip(lower=1)

        # 15. BathToRoomRatio (Bathrooms per room)
        df_eng['BathToRoomRatio'] = df_eng['TotalBath'] / tot_rooms.clip(lower=1)

        # 16. GarageSpacePerCar (Garage area divided by garage capacity)
        cars = get_col('GarageCars')
        df_eng['GarageSpacePerCar'] = get_col('GarageArea') / cars.clip(lower=1)

        # 17. OverallGrade (Simple sum of quality and condition)
        df_eng['OverallGrade'] = get_col('OverallQual', 5) + get_col('OverallCond', 5)

        # 18. BsmtFinSF (Total finished basement area)
        df_eng['BsmtFinSF'] = get_col('BsmtFinSF1') + get_col('BsmtFinSF2')

        # 19. BsmtUnfRatio (Unfinished basement ratio)
        bsmt_tot = get_col('TotalBsmtSF').clip(lower=1)
        df_eng['BsmtUnfRatio'] = get_col('BsmtUnfSF') / bsmt_tot

        # 20. HighQualFlrSF (High quality floor area - 1st floor vs 2nd floor)
        df_eng['HighQualFlrSF'] = get_col('1stFlrSF') + get_col('2ndFlrSF')

        # 21. LogLotArea (Log transform to reduce skewness)
        df_eng['LogLotArea'] = np.log1p(get_col('LotArea'))

        # 22. LogGrLivArea (Log transform of living area)
        df_eng['LogGrLivArea'] = np.log1p(get_col('GrLivArea'))

        return df_eng
