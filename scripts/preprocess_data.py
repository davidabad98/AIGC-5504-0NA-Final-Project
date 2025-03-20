import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the project root to the path so we can import project modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.data.preprocessing import ChurnDataPreprocessor


def load_config():
    """Load data preprocessing configuration from JSON file"""
    config_path = project_root / "configs" / "data_config.json"
    with open(config_path, "r") as file:
        return json.load(file)


def main():
    # Load configuration
    config = load_config()

    # Set paths
    raw_data_path = project_root / "data" / "raw" / config["raw_file_name"]
    processed_data_path = project_root / "data" / "processed"
    processed_file_name = config["processed_file_name"]
    preprocessor_path = processed_data_path / "preprocessor.pkl"

    # Create directories if they don't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)

    # Load the data
    print(f"Loading data from {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"Loaded data with shape: {df.shape}")

    # Create and fit the preprocessor
    target_column = config.get("target_column", "Churn")
    exclude_columns = config.get("exclude_columns", ["CustomerID"])

    print(f"Initializing preprocessor with target: {target_column}")
    preprocessor = ChurnDataPreprocessor(
        target=target_column, exclude_cols=exclude_columns
    )

    # Handle outliers if configured
    if config.get("remove_outliers", False):
        outlier_method = config.get("outlier_method", "IQR")
        outlier_threshold = config.get("outlier_threshold", 1.5)
        print(
            f"Removing outliers using {outlier_method} method with threshold {outlier_threshold}"
        )
        df = preprocessor.remove_outliers(
            df, method=outlier_method, threshold=outlier_threshold
        )
        print(f"Data shape after outlier removal: {df.shape}")

    # Fit the preprocessor to the cleaned data
    X, y = preprocessor.fit_transform(df)
    print(f"Fitted and transformed data")

    # Save the fitted preprocessor so that it can be reused later without refitting
    preprocessor.save_preprocessor(preprocessor_path)
    print(f"Saved preprocessor to: {preprocessor_path}")

    # (Optional) Save the processed data
    preprocessor.save_processed_data(X, y, processed_data_path, processed_file_name)

    # Prepare train/validation/test splits
    val_size = config.get("validation_size", 0.15)
    random_state = config.get("random_state", 42)

    print(f"Preparing data splits with val_size={val_size}")
    data_splits = preprocessor.prepare_train_val_test_split(
        df, val_size=val_size, random_state=random_state
    )

    splits = {
        "train": (data_splits["X_train"], data_splits["y_train"]),
        "val": (data_splits["X_val"], data_splits["y_val"]),
    }

    # Save processed data splits
    for split_name, (features, targets) in splits.items():
        # Save the processed data
        np.savez(
            processed_data_path / f"{split_name}_data.npz",
            features=features,
            targets=targets,
        )
        print(f"Saved {split_name} split with {features.shape[0]} samples")

    print("Data preprocessing completed successfully!")


if __name__ == "__main__":
    main()
