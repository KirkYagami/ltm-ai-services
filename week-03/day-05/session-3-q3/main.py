import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


NUMERICAL_COLUMNS = [
    "StudyHoursPerWeek",
    "AttendanceRate",
    "AssignmentScore",
    "QuizScore",
    "PreviousGPA",
]

CATEGORICAL_COLUMNS = [
    "CourseLevel",
    "LearningMode",
]

TARGET_COLUMN = "Passed"


def preprocess_data(filepath):
    """Load, preprocess, save, and split the student dataset."""

    raw_data = pd.read_csv(filepath)

    print("Dataset Shape:", raw_data.shape)
    print("\nMissing Values Before Cleaning:")
    print(raw_data.isnull().sum())

    X = raw_data.drop(columns=[TARGET_COLUMN])
    y = raw_data[TARGET_COLUMN]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()

    cleaned_data = pd.DataFrame(
        X_transformed,
        columns=feature_names,
        index=raw_data.index,
    )

    cleaned_data[TARGET_COLUMN] = y.values

    print(
        "\nMissing Values After Cleaning:",
        cleaned_data.isnull().sum().sum(),
    )

    cleaned_data.to_csv(
        "cleaned_student_data.csv",
        index=False,
    )

    print(
        "Cleaned data saved to cleaned_student_data.csv"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=0.20,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, raw_data


def train_model(X_train, y_train):
    """Train SGD classifiers with and without early stopping."""

    sgd_no_early_stop = SGDClassifier(
        loss="log_loss",
        alpha=1e-10,
        max_iter=5000,
        tol=None,
        random_state=42,
    )

    sgd_early_stop = SGDClassifier(
        loss="log_loss",
        alpha=1e-10,
        max_iter=5000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )

    sgd_no_early_stop.fit(
        X_train,
        y_train,
    )

    sgd_early_stop.fit(
        X_train,
        y_train,
    )

    joblib.dump(
        sgd_no_early_stop,
        "sgd_no_early_stop_model.joblib",
    )

    joblib.dump(
        sgd_early_stop,
        "sgd_early_stop_model.joblib",
    )

    print(
        "\nModels saved: "
        "sgd_no_early_stop_model.joblib, "
        "sgd_early_stop_model.joblib"
    )

    return sgd_no_early_stop, sgd_early_stop


def evaluate_model(
    sgd_no_early_stop,
    sgd_early_stop,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Evaluate accuracy and iteration count of both models."""

    no_stop_train = accuracy_score(
        y_train,
        sgd_no_early_stop.predict(X_train),
    )

    no_stop_test = accuracy_score(
        y_test,
        sgd_no_early_stop.predict(X_test),
    )

    early_stop_train = accuracy_score(
        y_train,
        sgd_early_stop.predict(X_train),
    )

    early_stop_test = accuracy_score(
        y_test,
        sgd_early_stop.predict(X_test),
    )

    no_stop_iterations = sgd_no_early_stop.n_iter_
    early_stop_iterations = sgd_early_stop.n_iter_

    results = {
        "no_early_stop": {
            "train": no_stop_train,
            "test": no_stop_test,
            "iterations": no_stop_iterations,
        },
        "early_stop": {
            "train": early_stop_train,
            "test": early_stop_test,
            "iterations": early_stop_iterations,
        },
    }

    no_stop_gap = no_stop_train - no_stop_test
    early_stop_gap = early_stop_train - early_stop_test

    print("\nSGDClassifier Early Stopping Comparison")
    print("=" * 45)

    print("\nWithout Early Stopping (Overfitting Risk):")
    print(f"  Iterations Run: {no_stop_iterations}")
    print(f"  Train Accuracy: {no_stop_train:.4f}")
    print(f"  Test Accuracy:  {no_stop_test:.4f}")
    print(f"  Accuracy Gap:   {no_stop_gap:.4f}")

    print("\nWith Early Stopping (Better Generalization):")
    print(f"  Iterations Run: {early_stop_iterations}")
    print(f"  Train Accuracy: {early_stop_train:.4f}")
    print(f"  Test Accuracy:  {early_stop_test:.4f}")
    print(f"  Accuracy Gap:   {early_stop_gap:.4f}")

    return results


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, raw_data = preprocess_data(
        "student_outcome_data.csv"
    )

    sgd_no_early_stop, sgd_early_stop = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        sgd_no_early_stop,
        sgd_early_stop,
        X_train,
        y_train,
        X_test,
        y_test,
    )