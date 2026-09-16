import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def preprocess_data():
    """Preprocess loan data and create train-test splits."""

    df = pd.read_csv("loan_default.csv")

    target_column = "Defaulted"

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    numerical_columns.remove(target_column)

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Fill numerical missing values with mean
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill categorical missing values with mode
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Save cleaned data before encoding/scaling
    df.to_csv("cleaned_loan_data.csv", index=False)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False
    )

    # Standardize numerical features
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(
        X[numerical_columns]
    )

    joblib.dump(scaler, "scaler.pkl")

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train and save L1 and L2 Logistic Regression models."""

    l1_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        random_state=42
    )

    l2_model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=42
    )

    l1_model.fit(X_train, y_train)
    l2_model.fit(X_train, y_train)

    joblib.dump(l1_model, "l1_model.pkl")
    joblib.dump(l2_model, "l2_model.pkl")

    return l1_model, l2_model


def compare_models(l1_model, l2_model, X_test, y_test):
    """Compare L1 and L2 models."""

    l1_predictions = l1_model.predict(X_test)
    l2_predictions = l2_model.predict(X_test)

    l1_accuracy = accuracy_score(y_test, l1_predictions)
    l2_accuracy = accuracy_score(y_test, l2_predictions)

    l1_nonzero = np.count_nonzero(l1_model.coef_)
    l2_nonzero = np.count_nonzero(l2_model.coef_)

    total_features = X_test.shape[1]

    print("\nL1 vs L2 Regularization Comparison:")
    print(f"L1 (Lasso) Accuracy: {l1_accuracy:.4f}")
    print(f"L2 (Ridge) Accuracy: {l2_accuracy:.4f}")
    print(
        f"L1 Non-zero Coefficients: "
        f"{l1_nonzero} / {total_features}"
    )
    print(
        f"L2 Non-zero Coefficients: "
        f"{l2_nonzero} / {total_features}"
    )

    return l1_accuracy, l2_accuracy, l1_nonzero, l2_nonzero


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    l1_model, l2_model = train_models(
        X_train,
        y_train
    )

    compare_models(
        l1_model,
        l2_model,
        X_test,
        y_test
    )