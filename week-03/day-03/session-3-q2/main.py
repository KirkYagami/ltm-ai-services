import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# ---------------------------------
# File Names
# ---------------------------------

TRAIN_FILE = "bank_campaign_data.xlsx"
TEST_FILE = "new_customers.xlsx"

MODEL_FILE = "logistic_model.joblib"
OUTPUT_FILE = "predicted_subscriptions.xlsx"


# ---------------------------------
# Task 1: Data Loading & Preprocessing
# ---------------------------------

def load_and_preprocess(train_df, test_df):
    """
    Prepare training and prediction datasets.

    Returns:
        X_train_processed
        y_train
        X_test_processed
        original_test_df
    """

    # Work on copies so the original DataFrames are not modified
    train_df = train_df.copy()
    test_df = test_df.copy()

    # ---------------------------------
    # Target separation
    # ---------------------------------

    # Encode target:
    # Yes -> 1
    # No  -> 0
    y = train_df["Subscribed"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove target from training features
    X_train = train_df.drop(columns=["Subscribed"])

    # Keep original test data for final output
    original_test_df = test_df.copy()

    # ---------------------------------
    # Feature columns
    # ---------------------------------

    categorical_features = [
        "Job",
        "MaritalStatus",
        "Education",
        "Contact",
        "Month"
    ]

    numerical_features = [
        "Age",
        "Balance",
        "Day",
        "Campaign"
    ]

    # ---------------------------------
    # Missing value handling
    # ---------------------------------

    # Numerical columns -> mean
    for column in numerical_features:
        mean_value = X_train[column].mean()

        X_train[column] = X_train[column].fillna(mean_value)
        test_df[column] = test_df[column].fillna(mean_value)

    # Categorical columns -> mode
    for column in categorical_features:
        mode_value = X_train[column].mode()[0]

        X_train[column] = X_train[column].fillna(mode_value)
        test_df[column] = test_df[column].fillna(mode_value)

    # ---------------------------------
    # Preprocessing
    # ---------------------------------

    # Numerical:
    # StandardScaler
    #
    # Categorical:
    # OneHotEncoder
    #
    # handle_unknown="ignore" is important
    # because prediction data may contain a
    # category that was not present in training.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        sparse_threshold=1.0
    )

    # ---------------------------------
    # FIT ONLY ON TRAINING DATA
    # ---------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    # ---------------------------------
    # TRANSFORM TEST DATA
    # ---------------------------------

    X_test_processed = preprocessor.transform(test_df)

    return (
        X_train_processed,
        y,
        X_test_processed,
        original_test_df
    )


# ---------------------------------
# Task 2: Model Training
# ---------------------------------

def train_classifier(X, y):
    """
    Train a Logistic Regression classifier.
    """

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X, y)

    return model


# ---------------------------------
# Task 3: Subscription Prediction
# ---------------------------------

def predict_subscription(
    model,
    processed_test_data,
    original_test_df
):
    """
    Generate subscription predictions and
    append them to the original test DataFrame.

    1 -> Yes
    0 -> No
    """

    # Make predictions
    predictions = model.predict(processed_test_data)

    # Convert numeric predictions to labels
    prediction_labels = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    # Work on a copy
    result_df = original_test_df.copy()

    # Add prediction column
    result_df["SubscriptionPrediction"] = prediction_labels

    return result_df


# ---------------------------------
# Model Persistence
# ---------------------------------

def save_model(model, path="logistic_model.joblib"):
    """
    Save trained model using Joblib.
    """

    joblib.dump(model, path)


def load_model(path="logistic_model.joblib"):
    """
    Load trained model from Joblib.
    """

    return joblib.load(path)


# ---------------------------------
# Main Workflow
# ---------------------------------

def main():

    # ---------------------------------
    # 1. Load datasets
    # ---------------------------------

    train_df = pd.read_excel(TRAIN_FILE)
    test_df = pd.read_excel(TEST_FILE)

    # ---------------------------------
    # 2. Preprocess
    # ---------------------------------

    X_train, y_train, X_test, original_test_df = (
        load_and_preprocess(
            train_df,
            test_df
        )
    )

    # ---------------------------------
    # 3. Train classifier
    # ---------------------------------

    model = train_classifier(
        X_train,
        y_train
    )

    # ---------------------------------
    # 4. Save model
    # ---------------------------------

    save_model(
        model,
        MODEL_FILE
    )

    # ---------------------------------
    # 5. Load model
    # ---------------------------------

    loaded_model = load_model(
        MODEL_FILE
    )

    # ---------------------------------
    # 6. Predict using LOADED model
    # ---------------------------------

    result_df = predict_subscription(
        loaded_model,
        X_test,
        original_test_df
    )

    # ---------------------------------
    # 7. Save predictions
    # ---------------------------------

    result_df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    # ---------------------------------
    # 8. Console output
    # ---------------------------------

    print("Top 10 Predicted Subscriptions:")
    print()

    print(result_df.head(10))


# ---------------------------------
# Program Entry Point
# ---------------------------------

if __name__ == "__main__":
    main()