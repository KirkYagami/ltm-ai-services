import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocess_data(filepath):
    """Preprocess patient data and create train-test sets."""

    raw_data = pd.read_csv(filepath)

    numerical_columns = [
        "Age",
        "BMI",
        "BloodPressure",
        "Cholesterol",
        "HeartRate"
    ]

    categorical_columns = [
        "SmokingStatus",
        "ExerciseFrequency"
    ]

    X = raw_data.drop(columns=["AtRisk"])
    y = raw_data["AtRisk"]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    X_processed = preprocessor.fit_transform(X)

    # Create fully processed dataframe for saving
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(X_processed, "toarray"):
        X_processed_for_csv = X_processed.toarray()
    else:
        X_processed_for_csv = X_processed

    cleaned_data = pd.DataFrame(
        X_processed_for_csv,
        columns=feature_names
    )

    cleaned_data["AtRisk"] = y.reset_index(drop=True)

    cleaned_data.to_csv(
        "cleaned_patient_data.csv",
        index=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, raw_data


def train_model(X_train, y_train):
    """Train KNN models with k=1 and k=50."""

    knn_k1 = KNeighborsClassifier(n_neighbors=1)
    knn_k50 = KNeighborsClassifier(n_neighbors=50)

    knn_k1.fit(X_train, y_train)
    knn_k50.fit(X_train, y_train)

    joblib.dump(
        knn_k1,
        "knn_k1_model.joblib"
    )

    joblib.dump(
        knn_k50,
        "knn_k50_model.joblib"
    )

    return knn_k1, knn_k50


def evaluate_model(
    knn_k1,
    knn_k50,
    X_train,
    y_train,
    X_test,
    y_test
):
    """Evaluate training and testing accuracy."""

    k1_train_predictions = knn_k1.predict(X_train)
    k1_test_predictions = knn_k1.predict(X_test)

    k50_train_predictions = knn_k50.predict(X_train)
    k50_test_predictions = knn_k50.predict(X_test)

    k1_train_accuracy = accuracy_score(
        y_train,
        k1_train_predictions
    )

    k1_test_accuracy = accuracy_score(
        y_test,
        k1_test_predictions
    )

    k50_train_accuracy = accuracy_score(
        y_train,
        k50_train_predictions
    )

    k50_test_accuracy = accuracy_score(
        y_test,
        k50_test_predictions
    )

    results = {
        "k1": {
            "train": k1_train_accuracy,
            "test": k1_test_accuracy
        },
        "k50": {
            "train": k50_train_accuracy,
            "test": k50_test_accuracy
        }
    }

    return results


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, raw_data = preprocess_data(
        "patient_risk_data.csv"
    )

    print(f"\nDataset Shape: {raw_data.shape}")

    print("\nMissing Values Before Cleaning:")
    print(raw_data.isnull().sum())

    cleaned_data = pd.read_csv(
        "cleaned_patient_data.csv"
    )

    print(
        "\nMissing Values After Cleaning:",
        cleaned_data.isnull().sum().sum()
    )

    print(
        "Cleaned data saved to cleaned_patient_data.csv"
    )

    knn_k1, knn_k50 = train_model(
        X_train,
        y_train
    )

    print(
        "\nModels saved: "
        "knn_k1_model.joblib, "
        "knn_k50_model.joblib"
    )

    results = evaluate_model(
        knn_k1,
        knn_k50,
        X_train,
        y_train,
        X_test,
        y_test
    )

    k1_gap = (
        results["k1"]["train"]
        - results["k1"]["test"]
    )

    k50_gap = (
        results["k50"]["train"]
        - results["k50"]["test"]
    )

    print(
        "\nKNN Model Comparison: "
        "Overfitting vs Underfitting"
    )
    print("=" * 48)

    print("\nk=1 (Overfitting - High Variance):")
    print(
        f"  Train Accuracy: "
        f"{results['k1']['train']:.4f}"
    )
    print(
        f"  Test Accuracy:  "
        f"{results['k1']['test']:.4f}"
    )
    print(
        f"  Accuracy Gap:   "
        f"{k1_gap:.4f}"
    )

    print("\nk=50 (Underfitting - High Bias):")
    print(
        f"  Train Accuracy: "
        f"{results['k50']['train']:.4f}"
    )
    print(
        f"  Test Accuracy:  "
        f"{results['k50']['test']:.4f}"
    )
    print(
        f"  Accuracy Gap:   "
        f"{k50_gap:.4f}"
    )