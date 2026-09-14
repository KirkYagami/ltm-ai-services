import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


FILE_NAME = "student_exam_data.xlsx"
PRED_FILE = "student_exam_predictions.xlsx"
ENCODED_FILE = "student_exam_data_encoded.xlsx"
MODEL_FILE = "model.joblib"


# ---------------------------------
# Task 1: Data Loading
# ---------------------------------
def load_dataset(file_path):
    """
    Load an Excel dataset and return it as a pandas DataFrame.
    """
    return pd.read_excel(file_path)


# ---------------------------------
# Task 2: Data Preprocessing
# ---------------------------------
def preprocess_data(df, encoder=None, fit_encoder=True):
    """
    Handle missing values and encode PreviousGrade.

    Parameters:
        df: pandas DataFrame
        encoder: LabelEncoder, used during prediction
        fit_encoder: whether to fit a new encoder

    Returns:
        processed_df, encoder
    """

    # Work on a copy so the original DataFrame is not modified
    processed_df = df.copy()

    # Handle missing numerical values
    if "StudyHours" in processed_df.columns:
        processed_df["StudyHours"] = processed_df["StudyHours"].fillna(
            processed_df["StudyHours"].mean()
        )

    # Handle missing categorical values
    if "PreviousGrade" in processed_df.columns:
        processed_df["PreviousGrade"] = processed_df["PreviousGrade"].fillna(
            processed_df["PreviousGrade"].mode()[0]
        )

    # Encode PreviousGrade
    if "PreviousGrade" in processed_df.columns:

        if fit_encoder:
            # Training:
            # Learn the mapping from categories to numbers
            encoder = LabelEncoder()
            processed_df["PreviousGrade"] = encoder.fit_transform(
                processed_df["PreviousGrade"]
            )

        else:
            # Prediction:
            # Reuse the encoder learned from training data
            if encoder is None:
                raise ValueError(
                    "An existing encoder is required when fit_encoder=False."
                )

            processed_df["PreviousGrade"] = encoder.transform(
                processed_df["PreviousGrade"]
            )

    return processed_df, encoder


# ---------------------------------
# Task 3: Feature Selection &
# Train-Test Split
# ---------------------------------
def split_dataset(df):
    """
    Select features and target, then perform train-test split.

    X:
        StudyHours
        PreviousGrade

    y:
        ExamScore
    """

    X = df[["StudyHours", "PreviousGrade"]]
    y = df["ExamScore"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------
# Task 4: Model Training
# ---------------------------------
def train_model(X_train, y_train):
    """
    Train and return a Linear Regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# ---------------------------------
# Task 5: Model Persistence
# ---------------------------------
def save_model(model, path="model.joblib"):
    """
    Save the trained model using Joblib.
    """

    joblib.dump(model, path)


def load_model(path="model.joblib"):
    """
    Load a previously saved model.
    """

    return joblib.load(path)


# ---------------------------------
# Task 6: Prediction on New Data
# ---------------------------------
def predict_new_data(model, encoder, file_path):
    """
    Load new student data, preprocess it using the
    same encoder, make predictions, and return a DataFrame.
    """

    # Load prediction data
    df = load_dataset(file_path)

    # Preprocess using the already-fitted encoder
    processed_df, _ = preprocess_data(
        df,
        encoder=encoder,
        fit_encoder=False
    )

    # Select exactly the features used during training
    X_new = processed_df[["StudyHours", "PreviousGrade"]]

    # Generate predictions
    predictions = model.predict(X_new)

    # Round predictions to 2 decimal places
    predictions = predictions.round(2)

    # Create output DataFrame
    result_df = df.copy()

    # Requirement: create PredictedExamScore
    result_df["PredictedExamScore"] = predictions

    return result_df


# ---------------------------------
# Main Workflow
# ---------------------------------
def main():
    # -----------------------------
    # 1. Load training data
    # -----------------------------
    df = load_dataset(FILE_NAME)

    # -----------------------------
    # 2. Preprocess training data
    # -----------------------------
    processed_df, encoder = preprocess_data(df)

    # Save cleaned and encoded dataset
    processed_df.to_excel(ENCODED_FILE, index=False)

    # -----------------------------
    # 3. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = split_dataset(processed_df)

    # -----------------------------
    # 4. Train model
    # -----------------------------
    model = train_model(X_train, y_train)

    # -----------------------------
    # 5. Save model
    # -----------------------------
    save_model(model, MODEL_FILE)

    # -----------------------------
    # 6. Reload model
    # -----------------------------
    loaded_model = load_model(MODEL_FILE)

    # -----------------------------
    # 7. Predict new student data
    # -----------------------------
    predicted_df = predict_new_data(
        loaded_model,
        encoder,
        PRED_FILE
    )

    # -----------------------------
    # 8. Console output
    # -----------------------------
    print("Predicted Exam Scores:")
    print()

    # The required prediction output
    print(
        predicted_df[
            [
                "StudyHours",
                "PreviousGrade",
                "PredictedExamScore"
            ]
        ]
    )


if __name__ == "__main__":
    main()