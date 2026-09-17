import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


TARGET_COLUMN = "Attrition"


def preprocess_data(filepath):
    """Clean, encode, split, and scale employee data."""

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

    # One-hot encode categorical features
    features = pd.get_dummies(
        features,
        drop_first=True,
    )

    # 80/20 stratified split
    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_features
    )

    test_scaled = scaler.transform(
        test_features
    )

    # Convert back to DataFrames and preserve columns/index
    train_features = pd.DataFrame(
        train_scaled,
        columns=train_features.columns,
        index=train_features.index,
    )

    test_features = pd.DataFrame(
        test_scaled,
        columns=test_features.columns,
        index=test_features.index,
    )

    return (
        train_features,
        test_features,
        train_target,
        test_target,
    )


def train_models(train_features, train_target):
    """Train Decision Tree and Random Forest models."""

    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    decision_tree.fit(
        train_features,
        train_target
    )

    random_forest.fit(
        train_features,
        train_target
    )

    models = {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }

    return models


def evaluate_models(
    models,
    test_features,
    test_target,
):
    """Evaluate classification models."""

    results = {}

    for model_name, model in models.items():

        predictions = model.predict(
            test_features
        )

        results[model_name] = {
            "Accuracy": accuracy_score(
                test_target,
                predictions,
            ),
            "Precision": precision_score(
                test_target,
                predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                test_target,
                predictions,
                zero_division=0,
            ),
            "F1 Score": f1_score(
                test_target,
                predictions,
                zero_division=0,
            ),
        }

    return results


if __name__ == "__main__":

    filepath = "employee_attrition.csv"

    # Original dataset size
    data = pd.read_csv(filepath)

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = preprocess_data(filepath)

    print(
        f"\nDataset Shape: {len(data)} rows"
    )

    print(
        f"Training Set: "
        f"{len(train_features)} rows"
    )

    print(
        f"Test Set: "
        f"{len(test_features)} rows"
    )

    # Train models
    models = train_models(
        train_features,
        train_target,
    )

    # Evaluate models
    results = evaluate_models(
        models,
        test_features,
        test_target,
    )

    print("\nModel Evaluation:")

    for model_name, metrics in results.items():

        print(f"\n{model_name}:")

        for metric_name, value in metrics.items():
            print(
                f"  {metric_name}: {value:.4f}"
            )

    # Select highest F1 model
    best_model_name = max(
        results,
        key=lambda name: results[name]["F1 Score"],
    )

    best_f1 = results[best_model_name][
        "F1 Score"
    ]

    print(
        f"\nBest Model: {best_model_name} "
        f"(F1: {best_f1:.4f})"
    )

    # Save best model
    joblib.dump(
        models[best_model_name],
        "attrition_model.joblib",
    )

    print(
        "Model saved to attrition_model.joblib"
    )