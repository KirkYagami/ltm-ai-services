import pandas as pd

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


DROP_COLUMNS = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin"
]

CATEGORICAL_COLUMNS = [
    "Sex",
    "Embarked"
]


def preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    target = df["Survived"].copy() if "Survived" in df.columns else None

    if "Survived" in df.columns:
        df = df.drop(columns=["Survived"])

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    df = df.astype(float)

    if target is not None:
        cleaned = df.copy()
        cleaned["Survived"] = target
        cleaned.to_csv(
            "cleaned_titanic_data.csv",
            index=False
        )

    return df, target


def train_model(features, target, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    classifier = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        random_state=42
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = round(
        accuracy_score(y_test, predictions),
        4
    )

    report = classification_report(
        y_test,
        predictions
    )

    model_path = f"decision_tree_depth_{max_depth}.joblib"
    dump(classifier, model_path)

    return classifier, float(accuracy), report


def predict_passengers(classifier, feature_columns, filepath):
    original_data = pd.read_csv(filepath)

    df = original_data.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True
    )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    df = df.astype(float)

    predictions = classifier.predict(df)

    result = original_data.copy()

    result["Predicted_Survival"] = [
        "Survived" if prediction == 1
        else "Not Survived"
        for prediction in predictions
    ]

    result.to_csv(
        "predicted_passengers.csv",
        index=False
    )

    return result


if __name__ == "__main__":
    features, target = preprocess_data("titanic.csv")

    shallow_model, shallow_accuracy, shallow_report = train_model(
        features,
        target,
        max_depth=3
    )

    print(
        f"Shallow Tree (max_depth=3) Accuracy: "
        f"{shallow_accuracy:.4f}"
    )
    print("Classification Report:")
    print(shallow_report)

    deep_model, deep_accuracy, deep_report = train_model(
        features,
        target,
        max_depth=10
    )

    print(
        f"\nDeep Tree (max_depth=10) Accuracy: "
        f"{deep_accuracy:.4f}"
    )
    print("Classification Report:")
    print(deep_report)

    results = predict_passengers(
        deep_model,
        features.columns,
        "new_passengers.csv"
    )

    print("\nPredictions saved to predicted_passengers.csv")
    print(results)