import numpy as np
import pandas as pd


def inverse_transform(preprocessor, X_transformed: np.ndarray) -> pd.DataFrame:
    """
    Perform an inverse transformation on the processed synthetic data to recover the original feature space.

    This function splits the synthetic data into numerical and categorical parts based on the
    preprocessor configuration, applies the individual inverse transformations, and then
    concatenates the results into a single DataFrame with the original column order.

    Args:
        preprocessor: A fitted ChurnDataPreprocessor instance with attributes:
            - preprocessor: The ColumnTransformer used for transformation.
            - numerical_features: List of names of numerical features.
            - categorical_features: List of names of categorical features.
            - original_feature_order: The original order of the features before transformation.
        X_transformed (np.ndarray): The synthetic data in the processed feature space.

    Returns:
        pd.DataFrame: DataFrame containing the data in the original feature space.
    """
    # Determine the number of numerical features
    num_feature_count = len(preprocessor.numerical_features)

    # For the categorical part, we need to know how many columns resulted from one-hot encoding.
    # Get the fitted OneHotEncoder from the categorical transformer pipeline.
    cat_pipeline = preprocessor.preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_feature_count = sum(len(cat) for cat in onehot.categories_)

    # Split the transformed array into numerical and categorical parts.
    X_num_trans = X_transformed[:, :num_feature_count]
    X_cat_trans = X_transformed[
        :, num_feature_count : num_feature_count + cat_feature_count
    ]

    # Inverse transform the numerical features.
    # The numerical pipeline consists of an imputer followed by a scaler.
    # Here we assume that synthetic data do not need imputation reversal.
    num_pipeline = preprocessor.preprocessor.named_transformers_["num"]
    scaler = num_pipeline.named_steps["scaler"]
    # The StandardScaler supports inverse_transform.
    X_num_inv = scaler.inverse_transform(X_num_trans)

    # Inverse transform the categorical features.
    # The categorical pipeline consists of an imputer and onehot encoder.
    # We use the onehot encoder’s inverse_transform method.
    X_cat_inv = onehot.inverse_transform(X_cat_trans)

    # Create DataFrames from each inverse-transformed part.
    df_num = pd.DataFrame(X_num_inv, columns=preprocessor.numerical_features)
    df_cat = pd.DataFrame(X_cat_inv, columns=preprocessor.categorical_features)

    # Concatenate the numerical and categorical parts.
    # The original feature order (recorded during fit) should be a concatenation
    # of numerical features followed by categorical features.
    df_original = pd.concat([df_num, df_cat], axis=1)
    # Reorder columns to match the original order saved during fit.
    df_original = df_original[preprocessor.original_feature_order]

    return df_original
