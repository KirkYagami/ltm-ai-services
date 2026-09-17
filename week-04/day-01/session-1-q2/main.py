import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


TARGET_COLUMN = "AQI_Category"


def preprocess_data(filepath):
    """Clean and standardize the air-quality data."""

    data = pd.read_csv(filepath)

    # Record missing values before imputation
    missing_before = data.isnull().sum()

    # Fill missing numerical values using median
    numerical_columns = data.select_dtypes(include="number").columns

    for column in numerical_columns:
        if column != TARGET_COLUMN:
            data[column] = data[column].fillna(
                data[column].median()
            )

    # Separate features and target
    target = data[TARGET_COLUMN].copy()
    features = data.drop(columns=[TARGET_COLUMN])

    # Standardize features
    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(features)

    # Convert back to DataFrame with original column names
    scaled_features = pd.DataFrame(
        scaled_array,
        columns=features.columns,
        index=features.index,
    )

    return scaled_features, target, missing_before


def apply_pca(scaled_features, variance_threshold=0.95):
    """Apply PCA while retaining the requested variance."""

    # Fit PCA using all available components
    pca_full = PCA()
    pca_full.fit(scaled_features)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    # Find minimum components required to reach threshold
    num_components = (
        np.searchsorted(
            cumulative_variance,
            variance_threshold,
        )
        + 1
    )

    # Fit PCA again using only required components
    pca_selected = PCA(n_components=num_components)

    reduced_array = pca_selected.fit_transform(
        scaled_features
    )

    # Create PC1, PC2, PC3, ...
    component_names = [
        f"PC{i}"
        for i in range(1, num_components + 1)
    ]

    reduced_df = pd.DataFrame(
        reduced_array,
        columns=component_names,
        index=scaled_features.index,
    )

    return reduced_df, pca_full, num_components


if __name__ == "__main__":

    scaled_features, target, missing_before = preprocess_data(
        "air_quality_sensors.csv"
    )

    print(
        f"\nDataset Shape: "
        f"{len(scaled_features)} rows, "
        f"{scaled_features.shape[1]} features"
    )

    print("\nMissing Values Before Imputation:")

    for column, count in missing_before.items():
        if count > 0:
            print(f"  {column}: {count}")

    print(
        "\nMissing Values After Imputation:",
        scaled_features.isnull().sum().sum()
        + target.isnull().sum(),
    )

    reduced_df, pca_full, num_components = apply_pca(
        scaled_features,
        variance_threshold=0.95,
    )

    print("\nPCA Results:")
    print(
        f"  Original Features: "
        f"{scaled_features.shape[1]}"
    )
    print(
        f"  Components for 95% Variance: "
        f"{num_components}"
    )

    print("\nExplained Variance per Component:")

    cumulative_variance = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    for i, (variance, cumulative) in enumerate(
        zip(
            pca_full.explained_variance_ratio_,
            cumulative_variance,
        ),
        start=1,
    ):
        print(
            f"  PC{i}: {variance:.4f} "
            f"(Cumulative: {cumulative:.4f})"
        )

    print(
        f"\nReduced Feature Matrix: "
        f"{scaled_features.shape} -> "
        f"{reduced_df.shape}"
    )