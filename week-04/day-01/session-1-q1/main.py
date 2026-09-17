import pandas as pd


TARGET_COLUMN = "FinalGrade"


def preprocess_data(filepath):
    """Clean the student data and encode categorical features."""

    data = pd.read_csv(filepath)

    missing_before = data.isnull().sum()

    numerical_columns = data.select_dtypes(
        include="number"
    ).columns.drop(TARGET_COLUMN)

    categorical_columns = data.select_dtypes(
        include=["object", "category"]
    ).columns

    # Median imputation for numerical features
    for column in numerical_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    # Mode imputation for categorical features
    for column in categorical_columns:
        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

    # Separate target before encoding
    target = data[TARGET_COLUMN].copy()
    features = data.drop(columns=[TARGET_COLUMN])

    # One-hot encode categorical features
    features = pd.get_dummies(
        features,
        columns=categorical_columns.tolist(),
        drop_first=True,
        dtype=int,
    )

    return features, target, missing_before


def select_features(features, target, top_k=5):
    """Select the top-k features by absolute target correlation."""

    correlations = features.corrwith(target).abs()

    correlations = correlations.sort_values(
        ascending=False
    )

    top_features = correlations.head(top_k).index

    selected_features = features[top_features].copy()

    return selected_features, correlations


if __name__ == "__main__":

    features, target, missing_before = preprocess_data(
        "student_performance.csv"
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

    selected_features, correlations = select_features(
        features,
        target,
        top_k=5,
    )

    print(
        "\nFeature Correlations with "
        "FinalGrade (all features):"
    )

    for feature, correlation in correlations.items():
        print(f"  {feature}: {correlation:.4f}")

    print("\nTop 5 Selected Features:")

    for feature in selected_features.columns:
        print(f"  {feature}")

    print(
        f"\nFeature Matrix: "
        f"{features.shape} -> "
        f"{selected_features.shape}"
    )