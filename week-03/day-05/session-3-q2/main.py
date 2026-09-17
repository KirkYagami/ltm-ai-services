import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


NUMERICAL_COLUMNS = [
    "AnnualIncome",
    "CreditScore",
    "LoanAmount",
    "EmploymentYears",
    "NumLatePayments",
]

CATEGORICAL_COLUMNS = [
    "LoanPurpose",
    "HomeOwnership",
]

TARGET_COLUMN = "Defaulted"


def preprocess_data(filepath):
    """Load, clean, transform, save, and split the credit-risk dataset."""

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
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
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

    print("\nMissing Values After Cleaning:",
          cleaned_data.isnull().sum().sum())

    cleaned_data.to_csv("cleaned_credit_data.csv", index=False)

    print("Cleaned data saved to cleaned_credit_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=0.20,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test, raw_data


def train_model(X_train, y_train):
    """Train RBF SVM models using weak and strong regularization."""

    svm_weak_reg = SVC(
        kernel="rbf",
        C=100,
        random_state=42,
    )

    svm_strong_reg = SVC(
        kernel="rbf",
        C=0.01,
        random_state=42,
    )

    svm_weak_reg.fit(X_train, y_train)
    svm_strong_reg.fit(X_train, y_train)

    joblib.dump(
        svm_weak_reg,
        "svm_c100_model.joblib",
    )

    joblib.dump(
        svm_strong_reg,
        "svm_c001_model.joblib",
    )

    print(
        "\nModels saved: "
        "svm_c100_model.joblib, "
        "svm_c001_model.joblib"
    )

    return svm_weak_reg, svm_strong_reg


def evaluate_model(
    svm_weak_reg,
    svm_strong_reg,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Compare training and testing accuracy of both SVM models."""

    weak_train_accuracy = accuracy_score(
        y_train,
        svm_weak_reg.predict(X_train),
    )

    weak_test_accuracy = accuracy_score(
        y_test,
        svm_weak_reg.predict(X_test),
    )

    strong_train_accuracy = accuracy_score(
        y_train,
        svm_strong_reg.predict(X_train),
    )

    strong_test_accuracy = accuracy_score(
        y_test,
        svm_strong_reg.predict(X_test),
    )

    results = {
        "weak_reg": {
            "train": weak_train_accuracy,
            "test": weak_test_accuracy,
        },
        "strong_reg": {
            "train": strong_train_accuracy,
            "test": strong_test_accuracy,
        },
    }

    weak_gap = abs(
        weak_train_accuracy - weak_test_accuracy
    )

    strong_gap = abs(
        strong_train_accuracy - strong_test_accuracy
    )

    print("\nSVM RBF Regularization Comparison")
    print("=" * 45)

    print("\nC=100 (Weak Regularization - Overfitting):")
    print(f"  Train Accuracy: {weak_train_accuracy:.4f}")
    print(f"  Test Accuracy:  {weak_test_accuracy:.4f}")
    print(f"  Accuracy Gap:   {weak_gap:.4f}")

    print("\nC=0.01 (Strong Regularization - Better Generalization):")
    print(f"  Train Accuracy: {strong_train_accuracy:.4f}")
    print(f"  Test Accuracy:  {strong_test_accuracy:.4f}")
    print(f"  Accuracy Gap:   {strong_gap:.4f}")

    return results


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, raw_data = preprocess_data(
        "credit_risk_data.csv"
    )

    svm_weak_reg, svm_strong_reg = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        svm_weak_reg,
        svm_strong_reg,
        X_train,
        y_train,
        X_test,
        y_test,
    )