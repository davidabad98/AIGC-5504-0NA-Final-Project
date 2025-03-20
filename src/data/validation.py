# src/data/validation.py
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/validation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validator for data quality checks.

    This class performs validation checks on the data including:
    - Missing values
    - Duplicates
    - Data type consistency
    - Value range validation
    - Statistical checks

    Attributes:
        validation_results (dict): Dictionary storing validation results
    """

    def __init__(self):
        """Initialize the validator."""
        self.validation_results = {}

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks on the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Dict[str, Any]: Dictionary with validation results
        """
        logger.info("Starting data validation")

        # Check for missing values
        self._check_missing_values(df)

        # Check for duplicates
        self._check_duplicates(df)

        # Check data types
        self._check_data_types(df)

        # Check value ranges
        self._check_value_ranges(df)

        # Check statistical properties
        self._check_statistical_properties(df)

        # Check target distribution
        self._check_target_distribution(df, "Churn")

        logger.info("Data validation complete")
        return self.validation_results

    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """
        Check for missing values in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100

        self.validation_results["missing_values"] = {
            "count": missing_values.to_dict(),
            "percentage": missing_percentage.to_dict(),
            "total_missing": missing_values.sum(),
            "passed": missing_values.sum() == 0,
        }

        if missing_values.sum() > 0:
            logger.warning(
                f"Found {missing_values.sum()} missing values in the dataset"
            )
        else:
            logger.info("No missing values found")

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """
        Check for duplicate rows in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        duplicates = df.duplicated().sum()

        self.validation_results["duplicates"] = {
            "count": duplicates,
            "percentage": (duplicates / len(df)) * 100,
            "passed": duplicates == 0,
        }

        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows in the dataset")
        else:
            logger.info("No duplicate rows found")

    def _check_data_types(self, df: pd.DataFrame) -> None:
        """
        Check data types of columns.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        dtypes = df.dtypes.astype(str).to_dict()

        expected_dtypes = {
            "CustomerID": ["float64", "int64"],
            "Age": ["float64", "int64"],
            "Gender": ["object"],
            "Tenure": ["float64", "int64"],
            "Usage Frequency": ["float64", "int64"],
            "Support Calls": ["float64", "int64"],
            "Payment Delay": ["float64", "int64"],
            "Subscription Type": ["object"],
            "Contract Length": ["object"],
            "Total Spend": ["float64"],
            "Last Interaction": ["float64", "int64"],
            "Churn": ["float64", "int64", "bool"],
        }

        type_issues = {}
        for column, expected_type_list in expected_dtypes.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type not in expected_type_list:
                    type_issues[column] = {
                        "expected": expected_type_list,
                        "actual": actual_type,
                    }

        self.validation_results["data_types"] = {
            "types": dtypes,
            "issues": type_issues,
            "passed": len(type_issues) == 0,
        }

        if len(type_issues) > 0:
            logger.warning(
                f"Found {len(type_issues)} columns with unexpected data types"
            )
        else:
            logger.info("All columns have expected data types")

    def _check_value_ranges(self, df: pd.DataFrame) -> None:
        """
        Check value ranges for numerical columns.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

        range_issues = {}
        for column in numerical_columns:
            min_val = df[column].min()
            max_val = df[column].max()

            # Define expected ranges for specific columns
            if column == "Age":
                if min_val < 0 or max_val > 120:
                    range_issues[column] = {
                        "min": min_val,
                        "max": max_val,
                        "expected_min": 0,
                        "expected_max": 120,
                    }
            elif column == "Tenure":
                if min_val < 0:
                    range_issues[column] = {
                        "min": min_val,
                        "max": max_val,
                        "expected_min": 0,
                        "expected_max": "unlimited",
                    }
            elif column == "Total Spend":
                if min_val < 0:
                    range_issues[column] = {
                        "min": min_val,
                        "max": max_val,
                        "expected_min": 0,
                        "expected_max": "unlimited",
                    }
            elif column == "Support Calls":
                if min_val < 0:
                    range_issues[column] = {
                        "min": min_val,
                        "max": max_val,
                        "expected_min": 0,
                        "expected_max": "unlimited",
                    }
            elif column == "Churn":
                # For binary classification, churn should be 0 or 1
                unique_values = df[column].unique()
                if not all(val in [0, 1] for val in unique_values):
                    range_issues[column] = {
                        "unique_values": list(unique_values),
                        "expected_values": [0, 1],
                    }

        self.validation_results["value_ranges"] = {
            "ranges": {
                col: {"min": df[col].min(), "max": df[col].max()}
                for col in numerical_columns
            },
            "issues": range_issues,
            "passed": len(range_issues) == 0,
        }

        if len(range_issues) > 0:
            logger.warning(f"Found {len(range_issues)} columns with value range issues")
        else:
            logger.info("All columns have expected value ranges")

    def _check_statistical_properties(self, df: pd.DataFrame) -> None:
        """
        Check statistical properties of numerical columns.

        Args:
            df (pd.DataFrame): Input dataframe
        """
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

        stats = {}
        for column in numerical_columns:
            stats[column] = {
                "mean": df[column].mean(),
                "median": df[column].median(),
                "std": df[column].std(),
                "skew": df[column].skew(),
                "kurtosis": df[column].kurtosis(),
            }

        self.validation_results["statistics"] = {
            "stats": stats,
        }

        logger.info("Statistical properties checked")

    def _check_target_distribution(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Check distribution of target variable.

        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
        """
        if target_column in df.columns:
            target_counts = df[target_column].value_counts().to_dict()

            # Calculate class imbalance
            if len(target_counts) > 1:
                majority_class = max(target_counts, key=target_counts.get)
                minority_class = min(target_counts, key=target_counts.get)
                imbalance_ratio = (
                    target_counts[majority_class] / target_counts[minority_class]
                )

                self.validation_results["target_distribution"] = {
                    "counts": target_counts,
                    "majority_class": majority_class,
                    "minority_class": minority_class,
                    "imbalance_ratio": imbalance_ratio,
                    "is_imbalanced": imbalance_ratio
                    > 3,  # Consider imbalanced if ratio > 3
                }

                if imbalance_ratio > 3:
                    logger.warning(
                        f"Target variable '{target_column}' is imbalanced with a ratio of {imbalance_ratio:.2f}"
                    )
                else:
                    logger.info(
                        f"Target variable '{target_column}' is well-balanced with a ratio of {imbalance_ratio:.2f}"
                    )
            else:
                self.validation_results["target_distribution"] = {
                    "counts": target_counts,
                    "is_imbalanced": False,
                }
                logger.warning(f"Target variable '{target_column}' has only one class")
        else:
            self.validation_results["target_distribution"] = {
                "error": f"Target column '{target_column}' not found in dataframe"
            }
            logger.error(f"Target column '{target_column}' not found in dataframe")

    def validate_synthetic_data(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate synthetic data against real data to ensure quality and similarity.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Dictionary with validation results
        """
        logger.info("Starting synthetic data validation")

        synthetic_validation = {}

        # Check column consistency
        synthetic_validation["column_consistency"] = self._check_column_consistency(
            real_df, synthetic_df
        )

        # Check data type consistency
        synthetic_validation["data_type_consistency"] = (
            self._check_data_type_consistency(real_df, synthetic_df)
        )

        # Check distribution similarity
        synthetic_validation["distribution_similarity"] = (
            self._check_distribution_similarity(real_df, synthetic_df)
        )

        # Check correlation structure
        synthetic_validation["correlation_structure"] = (
            self._check_correlation_structure(real_df, synthetic_df)
        )

        # Check privacy and identifiability
        synthetic_validation["privacy_check"] = self._check_privacy(
            real_df, synthetic_df
        )

        logger.info("Synthetic data validation complete")
        return synthetic_validation

    def _check_column_consistency(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check if synthetic data has the same columns as real data.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Column consistency check results
        """
        real_columns = set(real_df.columns)
        synthetic_columns = set(synthetic_df.columns)

        missing_columns = real_columns - synthetic_columns
        extra_columns = synthetic_columns - real_columns

        result = {
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns),
            "passed": len(missing_columns) == 0 and len(extra_columns) == 0,
        }

        if not result["passed"]:
            logger.warning(
                f"Column inconsistency detected: {len(missing_columns)} missing, {len(extra_columns)} extra"
            )
        else:
            logger.info("Column consistency check passed")

        return result

    def _check_data_type_consistency(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check if synthetic data has the same data types as real data.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Data type consistency check results
        """
        common_columns = list(set(real_df.columns) & set(synthetic_df.columns))

        type_inconsistencies = {}
        for column in common_columns:
            real_type = str(real_df[column].dtype)
            synthetic_type = str(synthetic_df[column].dtype)

            if real_type != synthetic_type:
                type_inconsistencies[column] = {
                    "real_type": real_type,
                    "synthetic_type": synthetic_type,
                }

        result = {
            "type_inconsistencies": type_inconsistencies,
            "passed": len(type_inconsistencies) == 0,
        }

        if not result["passed"]:
            logger.warning(
                f"Data type inconsistency detected in {len(type_inconsistencies)} columns"
            )
        else:
            logger.info("Data type consistency check passed")

        return result

    def _check_distribution_similarity(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check distribution similarity between real and synthetic data.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Distribution similarity check results
        """
        from scipy import stats

        common_columns = list(set(real_df.columns) & set(synthetic_df.columns))
        numerical_columns = [
            col for col in common_columns if real_df[col].dtype in ["int64", "float64"]
        ]
        categorical_columns = [
            col for col in common_columns if real_df[col].dtype == "object"
        ]

        # KS test for numerical features
        ks_test_results = {}
        for column in numerical_columns:
            # Filter out NaN values
            real_values = real_df[column].dropna().values
            synthetic_values = synthetic_df[column].dropna().values

            if len(real_values) > 0 and len(synthetic_values) > 0:
                statistic, p_value = stats.ks_2samp(real_values, synthetic_values)
                ks_test_results[column] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "similar_distribution": p_value
                    > 0.05,  # Using 0.05 significance level
                }

        # Chi-square test for categorical features
        chi2_test_results = {}
        for column in categorical_columns:
            real_counts = real_df[column].value_counts(normalize=True)
            synthetic_counts = synthetic_df[column].value_counts(normalize=True)

            # Align both series to have the same categories
            all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
            real_dist = np.array([real_counts.get(cat, 0) for cat in all_categories])
            synthetic_dist = np.array(
                [synthetic_counts.get(cat, 0) for cat in all_categories]
            )

            # Only perform test if there are enough unique values
            if len(all_categories) > 1:
                chi2, p_value = stats.chisquare(synthetic_dist, real_dist)
                chi2_test_results[column] = {
                    "chi2": chi2,
                    "p_value": p_value,
                    "similar_distribution": p_value
                    > 0.05,  # Using 0.05 significance level
                }

        # Overall distribution similarity
        numerical_similarity = (
            sum(result["similar_distribution"] for result in ks_test_results.values())
            / len(ks_test_results)
            if ks_test_results
            else 0
        )
        categorical_similarity = (
            sum(result["similar_distribution"] for result in chi2_test_results.values())
            / len(chi2_test_results)
            if chi2_test_results
            else 0
        )

        overall_similarity = (
            (
                (numerical_similarity * len(ks_test_results))
                + (categorical_similarity * len(chi2_test_results))
            )
            / (len(ks_test_results) + len(chi2_test_results))
            if (len(ks_test_results) + len(chi2_test_results)) > 0
            else 0
        )

        result = {
            "ks_test_results": ks_test_results,
            "chi2_test_results": chi2_test_results,
            "numerical_similarity": numerical_similarity,
            "categorical_similarity": categorical_similarity,
            "overall_similarity": overall_similarity,
            "passed": overall_similarity
            >= 0.7,  # Consider passed if at least 70% of columns have similar distributions
        }

        if not result["passed"]:
            logger.warning(
                f"Distribution similarity check failed with overall similarity of {overall_similarity:.2f}"
            )
        else:
            logger.info(
                f"Distribution similarity check passed with overall similarity of {overall_similarity:.2f}"
            )

        return result

    def _check_correlation_structure(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check correlation structure similarity between real and synthetic data.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Correlation structure check results
        """
        # Get numerical columns common to both dataframes
        common_columns = list(set(real_df.columns) & set(synthetic_df.columns))
        numerical_columns = [
            col for col in common_columns if real_df[col].dtype in ["int64", "float64"]
        ]

        if len(numerical_columns) < 2:
            logger.warning(
                "Not enough numerical columns to check correlation structure"
            )
            return {
                "error": "Not enough numerical columns to check correlation structure",
                "passed": False,
            }

        # Calculate correlation matrices
        real_corr = real_df[numerical_columns].corr().fillna(0)
        synthetic_corr = synthetic_df[numerical_columns].corr().fillna(0)

        # Calculate Frobenius norm of the difference
        diff_matrix = real_corr - synthetic_corr
        frobenius_norm = np.sqrt(np.sum(diff_matrix.values**2))

        # Calculate maximum possible Frobenius norm for normalization
        # For correlation matrices, max difference would be changing from -1 to 1 or vice versa
        max_frobenius_norm = np.sqrt(
            2 * len(numerical_columns) * (len(numerical_columns) - 1)
        )

        # Normalize to get similarity score
        if max_frobenius_norm > 0:
            similarity_score = 1 - (frobenius_norm / max_frobenius_norm)
        else:
            similarity_score = 1

        # Calculate difference in correlation for each pair
        pair_differences = {}
        for i, col1 in enumerate(numerical_columns):
            for j, col2 in enumerate(numerical_columns):
                if i < j:  # Only consider each pair once
                    real_val = real_corr.loc[col1, col2]
                    synthetic_val = synthetic_corr.loc[col1, col2]
                    abs_diff = abs(real_val - synthetic_val)

                    if abs_diff > 0.3:  # Flag pairs with correlation difference > 0.3
                        pair_differences[f"{col1}-{col2}"] = {
                            "real_correlation": real_val,
                            "synthetic_correlation": synthetic_val,
                            "absolute_difference": abs_diff,
                        }

        result = {
            "similarity_score": similarity_score,
            "frobenius_norm": frobenius_norm,
            "significant_differences": pair_differences,
            "passed": similarity_score
            >= 0.8,  # Consider passed if similarity score is at least 0.8
        }

        if not result["passed"]:
            logger.warning(
                f"Correlation structure check failed with similarity score of {similarity_score:.2f}"
            )
        else:
            logger.info(
                f"Correlation structure check passed with similarity score of {similarity_score:.2f}"
            )

        return result

    def _check_privacy(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Check privacy and identifiability of synthetic data compared to real data.

        Args:
            real_df (pd.DataFrame): Real dataframe
            synthetic_df (pd.DataFrame): Synthetic dataframe

        Returns:
            Dict[str, Any]: Privacy check results
        """

        # Calculate nearest neighbor distance ratio
        def calculate_distances(df1, df2, sample_size=1000):
            # Use only numerical columns
            common_columns = list(set(df1.columns) & set(df2.columns))
            numerical_columns = [
                col for col in common_columns if df1[col].dtype in ["int64", "float64"]
            ]

            if len(numerical_columns) == 0:
                return None

            # Sample rows if datasets are large
            if len(df1) > sample_size:
                df1_sample = df1[numerical_columns].sample(sample_size, random_state=42)
            else:
                df1_sample = df1[numerical_columns]

            if len(df2) > sample_size:
                df2_sample = df2[numerical_columns].sample(sample_size, random_state=42)
            else:
                df2_sample = df2[numerical_columns]

            # Normalize data
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            df1_normalized = pd.DataFrame(
                scaler.fit_transform(df1_sample), columns=numerical_columns
            )
            df2_normalized = pd.DataFrame(
                scaler.transform(df2_sample), columns=numerical_columns
            )

            # Calculate pairwise distances
            from sklearn.metrics import pairwise_distances

            return pairwise_distances(df1_normalized, df2_normalized)

        # Calculate minimum distances
        distances = calculate_distances(real_df, synthetic_df)

        if distances is None:
            logger.warning("No numerical columns available to check privacy")
            return {
                "error": "No numerical columns available to check privacy",
                "passed": False,
            }

        # Minimum distance from each real record to any synthetic record
        min_distances = np.min(distances, axis=1)
        avg_min_distance = np.mean(min_distances)

        # Define threshold for privacy concern
        privacy_threshold = 0.1  # This is a hyperparameter that may need tuning

        # Count records with privacy concerns
        privacy_concern_count = np.sum(min_distances < privacy_threshold)
        privacy_concern_percentage = (privacy_concern_count / len(min_distances)) * 100

        result = {
            "average_minimum_distance": avg_min_distance,
            "privacy_concern_count": privacy_concern_count,
            "privacy_concern_percentage": privacy_concern_percentage,
            "passed": privacy_concern_percentage
            <= 5,  # Consider passed if less than 5% of records have privacy concerns
        }

        if not result["passed"]:
            logger.warning(
                f"Privacy check failed with {privacy_concern_percentage:.2f}% of records having privacy concerns"
            )
        else:
            logger.info(
                f"Privacy check passed with {privacy_concern_percentage:.2f}% of records having privacy concerns"
            )

        return result

    def export_validation_results(self, output_path: str) -> None:
        """
        Export validation results to a JSON file.

        Args:
            output_path (str): Path to save the results
        """
        import json
        import os

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert numpy data types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            else:
                return obj

        # Convert validation results
        serializable_results = convert_numpy_types(self.validation_results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        logger.info(f"Validation results exported to {output_path}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a summary of the dataset.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            Dict[str, Any]: Summary statistics
        """
        summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "num_missing": df.isnull().sum().sum(),
            "num_duplicates": df.duplicated().sum(),
        }

        # Numerical columns summary
        numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
        summary["numerical_summary"] = {}

        for col in numerical_columns:
            summary["numerical_summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
            }

        # Categorical columns summary
        categorical_columns = df.select_dtypes(include=["object"]).columns
        summary["categorical_summary"] = {}

        for col in categorical_columns:
            value_counts = df[col].value_counts()
            top_values = value_counts.head(5).to_dict()
            num_unique = len(value_counts)

            summary["categorical_summary"][col] = {
                "num_unique": num_unique,
                "top_values": top_values,
            }

        # Target variable summary
        if "Churn" in df.columns:
            churn_distribution = df["Churn"].value_counts().to_dict()
            churn_percentage = (df["Churn"].sum() / len(df)) * 100

            summary["target_summary"] = {
                "distribution": churn_distribution,
                "churn_percentage": churn_percentage,
            }

        return summary

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a comprehensive human-readable validation report.

        This method creates a detailed Markdown report containing all validation results,
        including basic data quality checks and synthetic data validation if available.
        The report can be returned as a string and optionally saved to a file.

        Args:
            output_path (str, optional): Path to save the report as a Markdown file

        Returns:
            str: Complete validation report as a Markdown-formatted string
        """
        report = "# Data Validation Report\n\n"

        # Add report generation timestamp
        from datetime import datetime

        report += (
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Missing values
        report += "## Missing Values\n\n"
        if "missing_values" in self.validation_results:
            missing = self.validation_results["missing_values"]
            report += f"- Total missing values: {missing['total_missing']}\n"
            report += (
                f"- Status: {'✅ PASSED' if missing['passed'] else '❌ FAILED'}\n\n"
            )

            if not missing["passed"]:
                report += "### Columns with missing values\n\n"
                report += "| Column | Missing Count | Missing Percentage |\n"
                report += "| ------ | ------------- | ------------------ |\n"

                for col, count in missing["count"].items():
                    if count > 0:
                        pct = missing["percentage"][col]
                        report += f"| {col} | {count} | {pct:.2f}% |\n"

                report += "\n"

        # Duplicates
        report += "## Duplicate Rows\n\n"
        if "duplicates" in self.validation_results:
            dups = self.validation_results["duplicates"]
            report += f"- Duplicate rows: {dups['count']}\n"
            report += f"- Percentage: {dups['percentage']:.2f}%\n"
            report += f"- Status: {'✅ PASSED' if dups['passed'] else '❌ FAILED'}\n\n"

        # Data types
        report += "## Data Types\n\n"
        if "data_types" in self.validation_results:
            types = self.validation_results["data_types"]
            report += f"- Status: {'✅ PASSED' if types['passed'] else '❌ FAILED'}\n\n"

            report += "### Column Data Types\n\n"
            report += "| Column | Data Type |\n"
            report += "| ------ | --------- |\n"

            for col, dtype in types["types"].items():
                report += f"| {col} | {dtype} |\n"

            report += "\n"

            if not types["passed"]:
                report += "### Type Issues\n\n"
                report += "| Column | Expected Type | Actual Type |\n"
                report += "| ------ | ------------- | ----------- |\n"

                for col, issue in types["issues"].items():
                    expected = ", ".join(issue["expected"])
                    report += f"| {col} | {expected} | {issue['actual']} |\n"

                report += "\n"

        # Value ranges
        report += "## Value Ranges\n\n"
        if "value_ranges" in self.validation_results:
            ranges = self.validation_results["value_ranges"]
            report += (
                f"- Status: {'✅ PASSED' if ranges['passed'] else '❌ FAILED'}\n\n"
            )

            report += "### Numerical Ranges\n\n"
            report += "| Column | Min | Max |\n"
            report += "| ------ | --- | --- |\n"

            for col, range_info in ranges["ranges"].items():
                report += f"| {col} | {range_info['min']} | {range_info['max']} |\n"

            report += "\n"

            if not ranges["passed"]:
                report += "### Range Issues\n\n"
                report += "| Column | Current Range | Expected Range |\n"
                report += "| ------ | ------------- | -------------- |\n"

                for col, issue in ranges["issues"].items():
                    if "unique_values" in issue:
                        current = f"Values: {issue['unique_values']}"
                        expected = f"Values: {issue['expected_values']}"
                    else:
                        current = f"{issue['min']} to {issue['max']}"
                        expected = f"{issue['expected_min']} to {issue['expected_max']}"

                    report += f"| {col} | {current} | {expected} |\n"

                report += "\n"

        # Statistical properties
        report += "## Statistical Properties\n\n"
        if "statistics" in self.validation_results:
            stats = self.validation_results["statistics"]["stats"]

            report += "### Summary Statistics for Numerical Columns\n\n"
            report += "| Column | Mean | Median | Std Dev | Skewness | Kurtosis |\n"
            report += "| ------ | ---- | ------ | ------- | -------- | -------- |\n"

            for col, col_stats in stats.items():
                report += f"| {col} | {col_stats['mean']:.2f} | {col_stats['median']:.2f} | {col_stats['std']:.2f} | {col_stats['skew']:.2f} | {col_stats['kurtosis']:.2f} |\n"

            report += "\n"

        # Target distribution
        report += "## Target Distribution\n\n"
        if "target_distribution" in self.validation_results:
            target = self.validation_results["target_distribution"]

            if "error" in target:
                report += f"- Error: {target['error']}\n\n"
            else:
                if "counts" in target:
                    report += "### Class Distribution\n\n"
                    report += "| Class | Count | Percentage |\n"
                    report += "| ----- | ----- | ---------- |\n"

                    total = sum(target["counts"].values())
                    for cls, count in target["counts"].items():
                        percentage = (count / total) * 100
                        report += f"| {cls} | {count} | {percentage:.2f}% |\n"

                    report += "\n"

                if "is_imbalanced" in target:
                    imbalanced = target["is_imbalanced"]
                    report += f"- Class Imbalance: {'⚠️ IMBALANCED' if imbalanced else '✅ BALANCED'}\n"

                    if imbalanced and "imbalance_ratio" in target:
                        report += f"- Imbalance Ratio: {target['imbalance_ratio']:.2f} ({target['majority_class']} to {target['minority_class']})\n"
                        report += "- Note: Consider using techniques like oversampling, undersampling, or weighted loss functions to address class imbalance.\n\n"

        # Summary
        passed_checks = sum(
            1
            for key, value in self.validation_results.items()
            if isinstance(value, dict) and "passed" in value and value["passed"]
        )
        total_checks = sum(
            1
            for key, value in self.validation_results.items()
            if isinstance(value, dict) and "passed" in value
        )

        report += "# Validation Summary\n\n"
        report += f"- Passed Checks: {passed_checks}/{total_checks}\n"

        if passed_checks == total_checks:
            report += "- Overall Status: ✅ ALL CHECKS PASSED\n\n"
        else:
            report += "- Overall Status: ⚠️ SOME CHECKS FAILED\n\n"

        # Recommendations (provide specific recommendations based on failed checks)
        report += "# Recommendations\n\n"

        recommendations = []

        # Check for missing values
        if (
            "missing_values" in self.validation_results
            and not self.validation_results["missing_values"]["passed"]
        ):
            recommendations.append(
                "- Consider imputing missing values or removing columns with high missing percentages."
            )

        # Check for duplicates
        if (
            "duplicates" in self.validation_results
            and not self.validation_results["duplicates"]["passed"]
        ):
            recommendations.append(
                "- Remove duplicate records to avoid biasing the model."
            )

        # Check for data type issues
        if (
            "data_types" in self.validation_results
            and not self.validation_results["data_types"]["passed"]
        ):
            recommendations.append(
                "- Address data type inconsistencies by converting columns to appropriate types."
            )

        # Check for value range issues
        if (
            "value_ranges" in self.validation_results
            and not self.validation_results["value_ranges"]["passed"]
        ):
            recommendations.append(
                "- Investigate and correct out-of-range values that may indicate data quality issues."
            )

        # Check for imbalanced target
        if (
            "target_distribution" in self.validation_results
            and "is_imbalanced" in self.validation_results["target_distribution"]
            and self.validation_results["target_distribution"]["is_imbalanced"]
        ):
            recommendations.append(
                "- Address class imbalance by using oversampling, undersampling, or weighted loss functions."
            )

        # Add general recommendations if no specific issues found
        if not recommendations:
            recommendations.append(
                "- All validation checks passed. The data is ready for model training."
            )
        else:
            recommendations.append(
                "- After addressing the issues above, re-run validation to ensure data quality."
            )

        # Add recommendations to report
        for rec in recommendations:
            report += f"{rec}\n"

        # Save report if output_path provided
        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(report)
                logger.info(f"Validation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_path}: {str(e)}")

        return report
