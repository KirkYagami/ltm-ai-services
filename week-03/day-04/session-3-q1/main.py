import pandas as pd
from joblib import dump
from sklearn.svm import SVC


def preprocess_data(dataframe):
    """
    Preprocess customer churn data.

    - Fills missing numerical values with column medians.
    - Fills missing categorical values with column modes.
    - One-hot encodes categorical columns using drop_first=True.

    Returns:
        pandas.DataFrame: Cleaned and encoded dataframe.
    """
    df = dataframe.copy()

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Fill missing numerical values with median
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values with mode
    for column in categorical_columns:
        mode = df[column].mode()
        if not mode.empty:
            df[column] = df[column].fillna(mode.iloc[0])

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
    Train an SVC classification model.

    Parameters:
        cleaned_data: Preprocessed training dataframe.

    Returns:
        model: Trained SVC model.
        training_columns: Feature columns used during training.
    """
    # Separate target from features
    features = cleaned_data.drop(columns=["Churned"])
    target = cleaned_data["Churned"]

    # Store feature names for prediction
    training_columns = features.columns.tolist()

    # Initialize SVC with linear kernel
    model = SVC(
        kernel="linear",
        random_state=42
    )

    # Train model
    model.fit(features, target)

    # Save trained model
    dump(model, "svc_model.joblib")

    return model, training_columns


def predict_churn(model, prediction_data, training_columns):
    """
    Generate churn predictions for new customers.

    Parameters:
        model: Trained SVC model.
        prediction_data: Raw customer prediction data.
        training_columns: Feature columns used during training.

    Returns:
        pandas.DataFrame: Original prediction data with
        PredictedChurn column.
    """
    # Preserve original prediction data
    original_data = prediction_data.copy()

    # Preprocess prediction data
    cleaned_prediction = preprocess_data(prediction_data)

    # Align prediction columns with training columns
    cleaned_prediction = cleaned_prediction.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Generate predictions
    predictions = model.predict(cleaned_prediction)

    # Add predictions to original data
    result = original_data.copy()
    result["PredictedChurn"] = predictions

    # Save prediction results
    result.to_csv(
        "predicted_customers.csv",
        index=False
    )

    print("Predictions saved to 'predicted_customers.csv'")

    return result


if __name__ == "__main__":
    # Load training data
    training_data = pd.read_csv("churn_data.csv")

    # Load prediction data
    prediction_data = pd.read_csv("new_customers.csv")

    # Preprocess training data
    cleaned_data = preprocess_data(training_data)

    # Save cleaned training data
    cleaned_data.to_csv(
        "cleaned_churn_data.csv",
        index=False
    )

    print("Cleaned data saved as 'cleaned_churn_data.csv'")

    # Train SVC model
    model, training_columns = train_model(cleaned_data)

    print("Model saved as 'svc_model.joblib'")

    # Generate churn predictions
    results = predict_churn(
        model,
        prediction_data,
        training_columns
    )

    # Print sample predictions
    print()
    print("Sample Predictions:")
    print(results.head())