import pandas as pd
import joblib

from sklearn.svm import SVC


def preprocess_data(dataframe):
    """
    Preprocess and encode student performance data.

    - Identifies numerical and categorical columns.
    - Fills numerical missing values with the median.
    - Fills categorical missing values with the mode.
    - Applies one-hot encoding to categorical columns with drop_first=True.

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

    # Fill missing numerical values
    for column in numerical_columns:
        if column != "ExamResult":
            df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values
    for column in categorical_columns:
        mode_value = df[column].mode()
        if not mode_value.empty:
            df[column] = df[column].fillna(mode_value.iloc[0])

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
    Train an SVC model using a polynomial kernel.

    Returns:
        model: Trained SVC model.
        training_columns: Feature column names used during training.
    """
    # Separate target and features
    features = cleaned_data.drop(columns=["ExamResult"])
    target = cleaned_data["ExamResult"]

    # Store feature column names
    training_columns = features.columns.tolist()

    # Initialize SVC with polynomial kernel
    model = SVC(
        kernel="poly",
        random_state=42
    )

    # Train the model
    model.fit(features, target)

    # Save model
    joblib.dump(model, "svc_model.joblib")

    return model, training_columns


def predict_result(model, prediction_data, training_columns):
    """
    Generate exam outcome predictions for new students.

    Returns:
        pandas.DataFrame: Original prediction data with
        PredictedResult column added.
    """
    # Preserve original prediction data
    original_data = prediction_data.copy()

    # Preprocess prediction data
    processed_data = preprocess_data(prediction_data)

    # Align prediction columns with training columns
    processed_data = processed_data.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Generate predictions
    predictions = model.predict(processed_data)

    # Append predictions to original data
    original_data["PredictedResult"] = predictions

    # Save predictions
    original_data.to_csv(
        "predicted_students.csv",
        index=False
    )

    return original_data


if __name__ == "__main__":

    # Load training and prediction datasets
    student_data = pd.read_csv("student_performance.csv")
    new_students = pd.read_csv("new_students.csv")

    # Preprocess training data
    cleaned_data = preprocess_data(student_data)

    # Save cleaned training data
    cleaned_data.to_csv(
        "cleaned_student_data.csv",
        index=False
    )

    print("Cleaned data saved as 'cleaned_student_data.csv'")

    # Train model
    model, training_columns = train_model(cleaned_data)

    print("Model saved as 'svc_model.joblib'")

    # Generate predictions
    predicted_students = predict_result(
        model,
        new_students,
        training_columns
    )

    print("Predictions saved to 'predicted_students.csv'")
    print()
    print("Sample Predictions:")
    print(predicted_students.head(10))