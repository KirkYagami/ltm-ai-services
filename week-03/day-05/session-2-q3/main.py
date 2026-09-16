import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score


def preprocess_data():
    """Preprocess churn data and create train-test splits."""

    df = pd.read_csv("telecom_churn.csv")

    target_column = "Churned"

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
    df.to_csv("cleaned_churn_data.csv", index=False)

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

    # 80/20 stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """Train and save KNN and Naive Bayes models."""

    knn_model = KNeighborsClassifier(
        n_neighbors=5
    )

    nb_model = GaussianNB()

    knn_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)

    joblib.dump(knn_model, "knn_model.pkl")
    joblib.dump(nb_model, "nb_model.pkl")

    return knn_model, nb_model


def compare_models(knn_model, nb_model, X_test, y_test):
    """Compare models using F1 score."""

    knn_predictions = knn_model.predict(X_test)
    nb_predictions = nb_model.predict(X_test)

    knn_f1 = f1_score(
        y_test,
        knn_predictions
    )

    nb_f1 = f1_score(
        y_test,
        nb_predictions
    )

    if knn_f1 >= nb_f1:
        best_model_name = "KNN"
    else:
        best_model_name = "Naive Bayes"

    print("\nMulti-Model Comparison:")
    print(f"KNN F1 Score: {knn_f1:.4f}")
    print(f"Naive Bayes F1 Score: {nb_f1:.4f}")
    print(f"Best Model: {best_model_name}")

    return knn_f1, nb_f1, best_model_name


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    knn_model, nb_model = train_models(
        X_train,
        y_train
    )

    compare_models(
        knn_model,
        nb_model,
        X_test,
        y_test
    )