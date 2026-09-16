import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def preprocess_data():
    """Preprocess student data and create train-test splits."""

    df = pd.read_csv("student_performance.csv")

    target_column = "Passed"

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
    df.to_csv("cleaned_student_data.csv", index=False)

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


def tune_model(X_train, y_train):
    """Tune KNN using GridSearchCV."""

    model = KNeighborsClassifier()

    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy"
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    joblib.dump(best_model, "best_knn_model.pkl")

    return best_model, best_params, best_cv_score


def report_results(
    best_model,
    best_params,
    best_cv_score,
    X_test,
    y_test
):
    """Report GridSearchCV and test results."""

    predictions = best_model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nGridSearchCV Results:")
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {best_cv_score:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()

    best_model, best_params, best_cv_score = tune_model(
        X_train,
        y_train
    )

    report_results(
        best_model,
        best_params,
        best_cv_score,
        X_test,
        y_test
    )