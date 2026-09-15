import os
import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


NUMERICAL_COLUMNS = [
    "Age",
    "Annual_Income",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Marital_Status",
    "Occupation",
    "Education_Level",
]

TARGET_COLUMN = "Purchased"


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    for column in NUMERICAL_COLUMNS:
        df[column] = df[column].fillna(df[column].median())

    target = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    features = df.drop(columns=[TARGET_COLUMN])

    features = pd.get_dummies(
        features,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    features = features.astype(float)

    cleaned = features.copy()
    cleaned[TARGET_COLUMN] = target

    cleaned.to_csv(
        "cleaned_car_purchase_data.csv",
        index=False
    )

    return features, target


def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = round(
        accuracy_score(y_test, predictions),
        4
    )

    importances = pd.Series(
        classifier.feature_importances_,
        index=features.columns
    )

    top_features = (
        importances
        .sort_values(ascending=False)
        .head(3)
        .index
        .tolist()
    )

    dump(
        classifier,
        "decision_tree_model.joblib"
    )

    return classifier, float(accuracy), top_features


def predict_customers(classifier, feature_columns, filepath):
    new_data = pd.read_csv(filepath)

    for column in NUMERICAL_COLUMNS:
        new_data[column] = new_data[column].fillna(
            new_data[column].median()
        )

    encoded = pd.get_dummies(
        new_data,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    encoded = encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    encoded = encoded.astype(float)

    predictions = classifier.predict(encoded)

    result = new_data.copy()

    result["Predicted_Purchase"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    result.to_csv(
        "predicted_customers.csv",
        index=False
    )

    return result


if __name__ == "__main__":
    features, target = preprocess_data(
        "car_purchase_data.csv"
    )

    classifier, accuracy, top_features = train_model(
        features,
        target
    )

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Top 3 Features: {top_features}")

    results = predict_customers(
        classifier,
        features.columns,
        "new_customers.csv"
    )

    print("\nPredictions saved to predicted_customers.csv")
    print(results)