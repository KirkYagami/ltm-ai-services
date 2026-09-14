import os
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


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Task 1: Data Loading
# ---------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    """
    Load customer churn data from an Excel file.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    return pd.read_excel(path)


# ---------------------------------------------------------
# Task 2: Build Machine Learning Pipeline
# ---------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Build the complete scikit-learn preprocessing and
    Logistic Regression pipeline.

    Returns
    -------
    Pipeline
        Complete preprocessing + model pipeline.
    """

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="mean")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            ),
        ]
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(max_iter=2000)
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# Task 3: Feature & Target Preparation
# ---------------------------------------------------------

def split_features_target(df: pd.DataFrame):
    """
    Split dataframe into features and encoded target.

    Target encoding:
        Yes -> 1
        No  -> 0

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe.
    y : pd.Series
        Encoded target.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' not found."
        )

    # Verify required feature columns exist
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

    # Validate target values
    valid_targets = {"Yes", "No"}

    invalid_values = set(df[TARGET_COLUMN].dropna().unique()) - valid_targets

    if invalid_values:
        raise ValueError(
            f"Invalid target values found: {invalid_values}"
        )

    # Encode target
    y = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    # Ensure no missing target values
    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y


# ---------------------------------------------------------
# Task 4: Train & Evaluate Model
# ---------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, cv_folds=5):
    """
    Train and evaluate Logistic Regression model.

    Performs:
    - Feature/target separation
    - Stratified train/test split
    - Model training
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - Confusion matrix
    - Classification report
    - Cross-validation

    Returns
    -------
    pipeline : Pipeline
        Trained ML pipeline.

    metrics_dict : dict
        Evaluation metrics.

    cv_summary_dict : dict
        Cross-validation results.
    """

    X, y = split_features_target(df)

    # Stratified train/test split as required
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build pipeline
    pipeline = build_pipeline()

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # Cross-validation on the complete pipeline
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


# ---------------------------------------------------------
# Task 5: Model Persistence
# ---------------------------------------------------------

def save_model(model, path="churn_model.joblib"):
    """
    Save trained model/pipeline using joblib.
    """
    dump(model, path)


def load_model(path="churn_model.joblib"):
    """
    Load trained model/pipeline from joblib.
    """
    return load(path)


# ---------------------------------------------------------
# Task 6: Prediction on New Customers
# ---------------------------------------------------------

def predict_on_new(pipeline, new_df: pd.DataFrame):
    """
    Predict churn for new customers.

    Predictions:
        1 -> Yes
        0 -> No

    Adds:
        ChurnPrediction

    Returns
    -------
    pd.DataFrame
        Original dataframe with predictions.
    """

    required_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

    # Verify required columns exist
    missing_columns = [
        col for col in required_columns
        if col not in new_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    # Keep original dataframe unchanged
    result = new_df.copy()

    X_new = result[required_columns].copy()

    # Predict using the trained/saved pipeline
    predictions = pipeline.predict(X_new)

    # Convert 1/0 to Yes/No
    result["ChurnPrediction"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    return result


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------

if __name__ == "__main__":

    # Input files
    training_file = "customer_churn_data.xlsx"
    prediction_file = "new_customers.xlsx"

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    df = load_data(training_file)

    # -----------------------------------------------------
    # Train and evaluate
    # -----------------------------------------------------

    pipeline, metrics, cv_summary = train_and_evaluate(
        df,
        cv_folds=5
    )

    # -----------------------------------------------------
    # Print evaluation metrics
    # -----------------------------------------------------

    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    # -----------------------------------------------------
    # Print cross-validation results
    # -----------------------------------------------------

    print("\nCross-Validation Accuracy:")
    print(cv_summary["scores"])

    print(f"Mean CV Accuracy: {cv_summary['mean_accuracy']}")
    print(f"CV Std: {cv_summary['std_accuracy']}")

    # -----------------------------------------------------
    # Save trained model
    # -----------------------------------------------------

    model_path = "churn_model.joblib"

    save_model(
        pipeline,
        model_path
    )

    # -----------------------------------------------------
    # Load saved model
    # -----------------------------------------------------

    saved_model = load_model(model_path)

    # -----------------------------------------------------
    # Load new customer data
    # -----------------------------------------------------

    new_customers = load_data(prediction_file)

    # -----------------------------------------------------
    # Predict using SAVED model
    # -----------------------------------------------------

    predictions = predict_on_new(
        saved_model,
        new_customers
    )

    # -----------------------------------------------------
    # Export predictions
    # -----------------------------------------------------

    predictions.to_excel(
        "predicted_churn.xlsx",
        index=False
    )

    # -----------------------------------------------------
    # Print top 10 predictions
    # -----------------------------------------------------

    print("\nTop 10 Predictions:")
    print(predictions.head(10))