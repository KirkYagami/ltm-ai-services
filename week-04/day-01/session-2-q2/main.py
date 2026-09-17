import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


TARGET_COLUMN = "Defective"


def preprocess_data(filepath):
    """Preprocess the training dataset."""

    data = pd.read_csv(filepath)

    # Fill numerical missing values with median
    numerical_columns = data.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:
        if column != TARGET_COLUMN:
            data[column] = data[column].fillna(
                data[column].median()
            )

    # Fill categorical missing values with mode
    categorical_columns = data.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

    # Separate features and target
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]

    # One-hot encode categorical columns
    features = pd.get_dummies(
        features,
        drop_first=True
    )

    # Store encoded training columns
    feature_columns = features.columns.tolist()

    # Standardize features
    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(features)

    scaled_features = pd.DataFrame(
        scaled_values,
        columns=feature_columns,
        index=features.index
    )

    return (
        scaled_features,
        target,
        scaler,
        feature_columns
    )


def train_boosting(features, target):
    """Train Gradient Boosting classifier."""

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(features, target)

    return model


def predict_defects(
    model,
    filepath,
    scaler,
    feature_columns
):
    """Generate predictions for new production data."""

    data = pd.read_csv(filepath)

    # Keep original data for final output
    original_data = data.copy()

    # Fill numerical missing values with median
    numerical_columns = data.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:
        data[column] = data[column].fillna(
            data[column].median()
        )

    # Fill categorical missing values with mode
    categorical_columns = data.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        data[column] = data[column].fillna(
            data[column].mode()[0]
        )

    # Encode prediction data
    data = pd.get_dummies(
        data,
        drop_first=True
    )

    # Match prediction columns with training columns
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale using the scaler fitted on training data
    scaled_values = scaler.transform(data)

    scaled_data = pd.DataFrame(
        scaled_values,
        columns=feature_columns,
        index=data.index
    )

    # Generate predictions
    predictions = model.predict(scaled_data)

    # Append predictions to original prediction data
    original_data["PredictedDefective"] = predictions

    return original_data


if __name__ == "__main__":

    training_filepath = "product_quality.csv"
    prediction_filepath = "product_quality_predict.csv"

    # Load original data for reporting
    training_data = pd.read_csv(training_filepath)

    # Preprocess
    (
        features,
        target,
        scaler,
        feature_columns
    ) = preprocess_data(training_filepath)

    print(
        f"\nTraining Dataset: "
        f"{len(training_data)} rows, "
        f"{training_data.shape[1]} features"
    )

    # Train Gradient Boosting model
    model = train_boosting(
        features,
        target
    )

    print(
        "Gradient Boosting model trained successfully"
    )

    # Generate predictions
    result = predict_defects(
        model,
        prediction_filepath,
        scaler,
        feature_columns
    )

    # Save predictions
    result.to_csv(
        "predicted_defects.csv",
        index=False
    )

    print(
        f"\nPredictions saved to "
        f"predicted_defects.csv ({len(result)} rows)"
    )

    # Save trained model
    joblib.dump(
        model,
        "defect_model.joblib"
    )

    print(
        "Model saved to defect_model.joblib"
    )