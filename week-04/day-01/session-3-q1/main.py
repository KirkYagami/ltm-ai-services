import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TARGET_COLUMN = "Approved"


def preprocess_data(filepath):
    """Load, clean, and split the loan dataset."""

    data = pd.read_csv(filepath)

    # Fill numerical missing values with median
    numerical_columns = data.select_dtypes(include="number").columns

    for column in numerical_columns:
        if column != TARGET_COLUMN:
            data[column] = data[column].fillna(data[column].median())

    # Fill categorical missing values with mode
    categorical_columns = data.select_dtypes(include=["object"]).columns

    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    # Encode categorical columns if present
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and target
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]

    # 80/20 stratified split
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

    return train_features, test_features, train_target, test_target


def cap_outliers(features):
    """Detect and cap outliers using the IQR method."""

    capped = features.copy()

    outlier_counts = {}
    bounds = {}

    for column in features.columns:

        q1 = features[column].quantile(0.25)
        q3 = features[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # Count values outside IQR bounds
        outlier_count = (
            (features[column] < lower_bound) |
            (features[column] > upper_bound)
        ).sum()

        outlier_counts[column] = int(outlier_count)

        bounds[column] = (
            lower_bound,
            upper_bound
        )

        # Cap values
        capped[column] = features[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return capped, outlier_counts, bounds


def apply_bounds(features, bounds):
    """Apply training-set IQR bounds to another dataset."""

    capped = features.copy()

    for column in features.columns:

        lower_bound, upper_bound = bounds[column]

        capped[column] = features[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    return capped


def _calculate_metrics(target, predictions):
    """Calculate classification metrics."""

    return {
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(
            target,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            target,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            target,
            predictions,
            zero_division=0
        )
    }


def compare_boosting(
    raw_train,
    raw_test,
    capped_train,
    capped_test,
    train_target,
    test_target
):
    """Compare Gradient Boosting before and after outlier capping."""

    # --------------------------------------------------
    # Model 1: Before capping
    # --------------------------------------------------

    raw_scaler = StandardScaler()

    raw_train_scaled = raw_scaler.fit_transform(raw_train)
    raw_test_scaled = raw_scaler.transform(raw_test)

    raw_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    raw_model.fit(
        raw_train_scaled,
        train_target
    )

    raw_predictions = raw_model.predict(
        raw_test_scaled
    )

    before_metrics = _calculate_metrics(
        test_target,
        raw_predictions
    )

    # --------------------------------------------------
    # Model 2: After capping
    # --------------------------------------------------

    capped_scaler = StandardScaler()

    capped_train_scaled = capped_scaler.fit_transform(
        capped_train
    )

    capped_test_scaled = capped_scaler.transform(
        capped_test
    )

    capped_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    capped_model.fit(
        capped_train_scaled,
        train_target
    )

    capped_predictions = capped_model.predict(
        capped_test_scaled
    )

    after_metrics = _calculate_metrics(
        test_target,
        capped_predictions
    )

    comparison = {
        "Before Capping": before_metrics,
        "After Capping": after_metrics
    }

    return comparison, capped_model


if __name__ == "__main__":

    filepath = "loan_applications.csv"

    # Original dataset for reporting
    original_data = pd.read_csv(filepath)

    # Preprocess
    (
        train_features,
        test_features,
        train_target,
        test_target
    ) = preprocess_data(filepath)

    print(
        f"\nDataset Shape: {len(original_data)} rows"
    )

    print(
        f"Training Set: {len(train_features)} rows"
    )

    print(
        f"Test Set: {len(test_features)} rows"
    )

    # --------------------------------------------------
    # Detect/cap training outliers
    # --------------------------------------------------

    (
        capped_train,
        outlier_counts,
        bounds
    ) = cap_outliers(train_features)

    # Apply TRAINING bounds to test data
    capped_test = apply_bounds(
        test_features,
        bounds
    )

    print("\nOutliers Detected (IQR Method):")

    for column, count in outlier_counts.items():
        if count > 0:
            print(f"  {column}: {count}")

    # --------------------------------------------------
    # Compare models
    # --------------------------------------------------

    comparison, capped_model = compare_boosting(
        train_features,
        test_features,
        capped_train,
        capped_test,
        train_target,
        test_target
    )

    print(
        "\nGradient Boosting - "
        "Outlier Impact Comparison:"
    )

    for stage, metrics in comparison.items():

        print(f"\n{stage}:")

        for metric_name, value in metrics.items():
            print(
                f"  {metric_name}: {value:.4f}"
            )

    # Save model trained on capped data
    joblib.dump(
        capped_model,
        "approval_model.joblib"
    )

    print(
        "\nModel saved to approval_model.joblib"
    )