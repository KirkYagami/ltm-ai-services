import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def preprocess_data(dataframe):
    """
    Clean and encode the product inspection dataset.

    - Numerical missing values are replaced with the column median.
    - Categorical missing values are replaced with the column mode.
    - Categorical columns are one-hot encoded using drop_first=True.

    Returns:
        pandas.DataFrame: Cleaned and encoded dataframe.
    """
    df = dataframe.copy()

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Do not impute the target column as it is specified to have
    # no missing values.
    numerical_features = [
        col for col in numerical_columns
        if col != "DefectFound"
    ]

    # Fill missing numerical values with median
    for column in numerical_features:
        df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values with mode
    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode().iloc[0])

    # One-hot encode categorical columns
    if categorical_columns:
        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )

    return df


def train_model(cleaned_data):
    """
    Train an SVC model using an RBF kernel.

    The data is split into 80% training and 20% testing data.

    Returns:
        model: Trained SVC model.
        features_test: Test feature dataframe.
        target_test: Test target series.
    """
    # Separate target from features
    features = cleaned_data.drop(columns=["DefectFound"])
    target = cleaned_data["DefectFound"]

    # Split into 80% training and 20% testing
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    # Initialize SVC with RBF kernel
    model = SVC(
        kernel="rbf",
        random_state=42
    )

    # Train the model
    model.fit(features_train, target_train)

    # Save trained model
    joblib.dump(model, "svc_model.joblib")

    return model, features_test, target_test


def evaluate_model(model, features_test, target_test):
    """
    Evaluate the trained SVC model.

    Returns:
        accuracy: Accuracy score.
        f1: F1-score.
        report: Classification report as a string.
    """
    # Generate predictions
    predictions = model.predict(features_test)

    # Calculate metrics
    accuracy = accuracy_score(target_test, predictions)
    f1 = f1_score(target_test, predictions)

    # Generate classification report
    report = classification_report(
        target_test,
        predictions
    )

    # Save evaluation metrics
    evaluation_results = pd.DataFrame({
        "Accuracy": [accuracy],
        "F1-Score": [f1]
    })

    evaluation_results.to_csv(
        "evaluation_results.csv",
        index=False
    )

    return accuracy, f1, report


if __name__ == "__main__":

    # Load training data
    inspection_data = pd.read_csv("product_inspection.csv")

    # Preprocess data
    cleaned_data = preprocess_data(inspection_data)

    # Save cleaned data
    cleaned_data.to_csv(
        "cleaned_inspection_data.csv",
        index=False
    )

    print("Cleaned data saved as 'cleaned_inspection_data.csv'")

    # Train model
    model, features_test, target_test = train_model(cleaned_data)

    print("Model saved as 'svc_model.joblib'")

    # Evaluate model
    accuracy, f1, report = evaluate_model(
        model,
        features_test,
        target_test
    )

    print("Evaluation results saved to 'evaluation_results.csv'")
    print()
    print("Model Evaluation Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print()
    print("Classification Report:")
    print(report)