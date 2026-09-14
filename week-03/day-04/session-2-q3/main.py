import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(dataframe):
    """
    Clean and encode the ride dataset.

    - Fills missing numerical values with the median.
    - Fills missing categorical values with the mode.
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

    # Fill numerical missing values with median
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values with mode
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
    Train a Random Forest Regression model.

    Parameters:
        cleaned_data: Preprocessed training dataframe.

    Returns:
        model: Trained RandomForestRegressor.
        training_columns: Feature columns used during training.
    """
    # Separate target from features
    features = cleaned_data.drop(
        columns=["FareAmount"]
    )
    target = cleaned_data["FareAmount"]

    # Store feature columns for prediction
    training_columns = features.columns.tolist()

    # Initialize Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        bootstrap=True,
        random_state=42
    )

    # Train model
    model.fit(features, target)

    # Save model
    dump(model, "random_forest_model.joblib")

    return model, training_columns


def predict_fares(model, prediction_data, training_columns):
    """
    Generate fare predictions for new rides.

    Parameters:
        model: Trained RandomForestRegressor.
        prediction_data: Raw new ride dataframe.
        training_columns: Feature columns used during training.

    Returns:
        pandas.DataFrame: Original prediction data with PredictedFare.
    """
    # Keep original data unchanged for output
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

    # Append predictions to original data
    result = original_data.copy()
    result["PredictedFare"] = predictions.round(2)

    # Save predictions
    result.to_csv(
        "predicted_fares.csv",
        index=False
    )

    print("Predictions saved to 'predicted_fares.csv'")

    return result


if __name__ == "__main__":
    # Load training data
    training_data = pd.read_csv("ride_data.csv")

    # Load prediction data
    prediction_data = pd.read_csv("new_rides.csv")

    # Preprocess training data
    cleaned_data = preprocess_data(training_data)

    # Save cleaned training data
    cleaned_data.to_csv(
        "cleaned_ride_data.csv",
        index=False
    )

    print("Cleaned data saved as 'cleaned_ride_data.csv'")

    # Train model
    model, training_columns = train_model(cleaned_data)

    print("Model saved as 'random_forest_model.joblib'")

    # Generate predictions
    results = predict_fares(
        model,
        prediction_data,
        training_columns
    )

    # Print sample predictions
    print()
    print("Sample Predictions:")
    print(results.head())