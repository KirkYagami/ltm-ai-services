import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


TARGET_COLUMN = "HighSpender"


def preprocess_data(filepath):
    """Load data, impute missing values, and separate target."""

    data = pd.read_csv(filepath)

    # Record missing values before imputation
    missing_before = data.isnull().sum()

    # Fill missing numerical values with median
    numerical_columns = data.select_dtypes(include="number").columns

    for column in numerical_columns:
        if column != TARGET_COLUMN:
            data[column] = data[column].fillna(
                data[column].median()
            )

    # Separate features and target
    target = data[TARGET_COLUMN].copy()
    features = data.drop(columns=[TARGET_COLUMN]).copy()

    return features, target, missing_before


def detect_outliers(features):
    """Detect and cap outliers using the IQR method."""

    capped_features = features.copy()
    outlier_counts = {}

    numerical_columns = features.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        q1 = features[column].quantile(0.25)
        q3 = features[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Detect outliers before capping
        outlier_mask = (
            (features[column] < lower_bound)
            | (features[column] > upper_bound)
        )

        outlier_counts[column] = int(outlier_mask.sum())

        # Cap values at IQR boundaries
        capped_features[column] = features[column].clip(
            lower=lower_bound,
            upper=upper_bound,
        )

    return capped_features, outlier_counts


def compare_pca(raw_features, capped_features, top_n=3):
    """Compare PCA variance before and after outlier capping."""

    # Raw and capped datasets require separate scalers
    raw_scaler = StandardScaler()
    capped_scaler = StandardScaler()

    raw_scaled = raw_scaler.fit_transform(raw_features)
    capped_scaled = capped_scaler.fit_transform(
        capped_features
    )

    # Fit full PCA models
    pca_before = PCA()
    pca_after = PCA()

    pca_before.fit(raw_scaled)
    pca_after.fit(capped_scaled)

    # Extract top N explained variance ratios
    variance_before = (
        pca_before.explained_variance_ratio_[:top_n]
    )

    variance_after = (
        pca_after.explained_variance_ratio_[:top_n]
    )

    return variance_before, variance_after


if __name__ == "__main__":

    features, target, missing_before = preprocess_data(
        "ecommerce_transactions.csv"
    )

    print(
        f"\nDataset Shape: "
        f"{len(features)} rows, "
        f"{features.shape[1]} features"
    )

    print("\nMissing Values Before Imputation:")

    for column, count in missing_before.items():
        if count > 0:
            print(f"  {column}: {count}")

    print(
        "\nMissing Values After Imputation:",
        features.isnull().sum().sum()
        + target.isnull().sum(),
    )

    capped_features, outlier_counts = detect_outliers(
        features
    )

    print("\nOutliers Detected (IQR Method):")

    total_outliers = 0

    for column, count in outlier_counts.items():
        if count > 0:
            print(f"  {column}: {count}")
            total_outliers += count

    print(f"  Total: {total_outliers}")

    variance_before, variance_after = compare_pca(
        features,
        capped_features,
        top_n=3,
    )

    print(
        "\nPCA Variance Comparison "
        "(Top 3 Components):"
    )

    print(
        f"{'Component':<12}"
        f"{'Before Capping':<20}"
        f"{'After Capping':<20}"
    )

    for i in range(len(variance_before)):
        print(
            f"{f'PC{i + 1}':<12}"
            f"{variance_before[i]:<20.4f}"
            f"{variance_after[i]:<20.4f}"
        )