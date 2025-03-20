# src/data/preprocessing.py
import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "preprocessing.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ChurnDataPreprocessor:
    """
    Preprocessor for Customer Churn dataset.

    This class handles all data preprocessing steps including:
    - Data cleaning
    - Handling missing values
    - Encoding categorical variables
    - Scaling numerical features
    - Outlier detection and handling
    - Train-test split

    Attributes:
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        target (str): Name of the target variable
        preprocessor (ColumnTransformer): Sklearn preprocessor for feature transformation
        label_encoder (LabelEncoder): Encoder for the target variable if needed
    """

    def __init__(self, target: str = "Churn", exclude_cols: list = None):
        """
        Initialize the preprocessor.

        Args:
            target (str): Name of the target variable
            exclude_cols (list): List of columns to exclude from preprocessing
        """
        self.target = target
        self.exclude_cols = exclude_cols or ["CustomerID"]
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.label_encoder = None
        self.numerical_scaler = StandardScaler()
        self.feature_names = None

    def fit(self, df: pd.DataFrame) -> "ChurnDataPreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            ChurnDataPreprocessor: Self instance
        """
        logger.info("Starting preprocessing pipeline fitting")

        # Identify feature types
        self._identify_feature_types(df)

        # Create column transformer for preprocessing
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.numerical_features),
                ("cat", categorical_transformer, self.categorical_features),
            ],
            remainder="drop",
        )

        # Get feature names to use after transformation
        self.preprocessor.fit(df.drop(columns=[self.target] + self.exclude_cols))

        # Create feature names for transformed data
        self._create_feature_names()

        # If target is categorical, prepare label encoder
        if df[self.target].dtype == "object":
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df[self.target])

        logger.info("Preprocessing pipeline fitted successfully")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the data using the fitted preprocessor.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features and target
        """
        logger.info(f"Transforming data with shape {df.shape}")

        # Transform features
        X = self.preprocessor.transform(
            df.drop(columns=[self.target] + self.exclude_cols)
        )

        # Transform target if needed
        if self.label_encoder is not None:
            y = self.label_encoder.transform(df[self.target])
        else:
            y = df[self.target].values

        logger.info(f"Data transformed successfully. X shape: {X.shape}")
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor to the data and transform it.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed features and target
        """
        return self.fit(df).transform(df)

    def prepare_train_val_test_split(
        self,
        df: pd.DataFrame,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Prepare train, validation splits.

        Args:
            df (pd.DataFrame): Input dataframe
            val_size (float): Size of validation set
            test_size (float): Size of test set
            random_state (int): Random seed

        Returns:
            Dict[str, np.ndarray]: Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing train and validation")

        # First split: separate test set
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=random_state,
            stratify=df[self.target],
        )

        # Preprocess each split
        X_train, y_train = self.transform(train_df)
        X_val, y_val = self.transform(val_df)

        logger.info(f"Split complete. Train: {X_train.shape}, Val: {X_val.shape}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }

    def remove_outliers(
        self, df: pd.DataFrame, method: str = "IQR", threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from numerical features.

        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Method to use for outlier detection ('IQR' or 'zscore')
            threshold (float): Threshold for outlier detection

        Returns:
            pd.DataFrame: Dataframe with outliers removed
        """
        logger.info(f"Removing outliers using {method} method")
        df_clean = df.copy()

        # Ensure feature types are identified
        if self.numerical_features is None:
            self._identify_feature_types(df)

        if method == "IQR":
            for feature in self.numerical_features:
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[
                    (df_clean[feature] >= lower_bound)
                    & (df_clean[feature] <= upper_bound)
                ]

        elif method == "zscore":
            from scipy import stats

            for feature in self.numerical_features:
                z_scores = np.abs(stats.zscore(df_clean[feature]))
                df_clean = df_clean[z_scores < threshold]

        logger.info(
            f"Outlier removal complete. Rows removed: {len(df) - len(df_clean)}"
        )
        return df_clean

    def get_feature_names(self) -> list:
        """
        Get feature names after transformation.

        Returns:
            list: List of feature names
        """
        return self.feature_names

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Identify numerical and categorical features.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        # Exclude target and specified columns
        exclude = [self.target] + self.exclude_cols

        # Identify numerical and categorical features
        self.numerical_features = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # Remove excluded columns
        self.numerical_features = [
            col for col in self.numerical_features if col not in exclude
        ]
        self.categorical_features = [
            col for col in self.categorical_features if col not in exclude
        ]

        logger.info(
            f"Identified {len(self.numerical_features)} numerical features and {len(self.categorical_features)} categorical features"
        )

    def _create_feature_names(self) -> None:
        """
        Create feature names for transformed data.
        """
        numerical_names = self.numerical_features

        # Get one-hot encoded feature names
        ohe = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        categorical_names = []
        for i, feature in enumerate(self.categorical_features):
            feature_values = ohe.categories_[i]
            for value in feature_values:
                categorical_names.append(f"{feature}_{value}")

        self.feature_names = numerical_names + categorical_names
        logger.info(
            f"Created {len(self.feature_names)} feature names for transformed data"
        )

    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the preprocessor to disk.

        Args:
            filepath (str): Path to save the preprocessor
        """
        import joblib

        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath: str) -> "ChurnDataPreprocessor":
        """
        Load the preprocessor from disk.

        Args:
            filepath (str): Path to load the preprocessor from

        Returns:
            ChurnDataPreprocessor: Loaded preprocessor
        """
        import joblib

        logger.info(f"Loading preprocessor from {filepath}")
        return joblib.load(filepath)

    def save_processed_data(
        self, X: np.ndarray, y: np.ndarray, processed_dir: str, file_name: str
    ) -> None:
        """
        Reassemble the transformed features and target into a DataFrame and save as CSV.

        Args:
            X (np.ndarray): Transformed features
            y (np.ndarray): Transformed target
            processed_dir (str): Directory to save the processed data CSV
        """
        # Ensure the processed directory exists
        os.makedirs(processed_dir, exist_ok=True)

        # Get feature names
        feature_names = self.get_feature_names()

        # Reassemble into a DataFrame
        df_transformed = pd.DataFrame(X, columns=feature_names)
        df_transformed[self.target] = y

        # Define the output path
        processed_data_path = os.path.join(processed_dir, file_name)

        # Save the DataFrame to CSV
        df_transformed.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}")
