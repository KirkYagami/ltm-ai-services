import pandas as pd

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


NUMERICAL_COLUMNS = [
    "Age",
    "Tenure",
    "MonthlyCharges",
    "SupportCalls",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "ContractType",
    "InternetService",
]

TARGET_COLUMN = "Churn"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def build_pipeline() -> Pipeline:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    return pipeline


def split_features_target(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' not found."
        )

    required_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    X = df[required_columns].copy()

    valid_targets = {"Yes", "No"}
    invalid_values = set(
        df[TARGET_COLUMN].dropna().unique()
    ) - valid_targets

    if invalid_values:
        raise ValueError(
            f"Invalid target values found: {invalid_values}"
        )

    y = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y


def train_and_evaluate(df: pd.DataFrame, cv_folds=5):
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics_dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred
        ),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv_folds,
        scoring="accuracy"
    )

    cv_summary_dict = {
        "scores": cv_scores,
        "mean_accuracy": cv_scores.mean(),
        "std_accuracy": cv_scores.std(),
    }

    return pipeline, metrics_dict, cv_summary_dict


def save_model(model, path="churn_model.joblib"):
    dump(model, path)


def load_model(path="churn_model.joblib"):
    return load(path)


def predict_on_new(pipeline, new_df):
    required_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in new_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    result = new_df.copy()
    X_new = result[required_columns].copy()

    predictions = pipeline.predict(X_new)

    result["ChurnPrediction"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    return result


if __name__ == "__main__":
    training_file = "customer_churn_data.xlsx"
    prediction_file = "new_customers.xlsx"
    model_path = "churn_model.joblib"

    df = load_data(training_file)

    pipeline, metrics, cv_summary = train_and_evaluate(
        df,
        cv_folds=5
    )

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    save_model(pipeline, model_path)

    saved_model = load_model(model_path)

    new_customers = load_data(prediction_file)

    predictions = predict_on_new(
        saved_model,
        new_customers
    )

    predictions.to_excel(
        "predicted_churn.xlsx",
        index=False
    )

    print("\nTop 10 Predictions:")
    print(predictions.head(10))