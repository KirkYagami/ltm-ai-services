import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor


def preprocess_data(dataframe):
    """
    Clean and encode the insurance claims dataset.

    Removes PolicyID, handles missing values, and one-hot encodes
    categorical columns.
    """
    df = dataframe.copy()

    # PolicyID is a non-predictive identifier
    df = df.drop(columns=["PolicyID"], errors="ignore")

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Fill numerical missing values with column median
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].median())

    # Fill categorical missing values with column mode
    for column in categorical_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])

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
    Train a Random Forest Regression model using ClaimCost as target.

    Returns:
        model: Trained RandomForestRegressor
        training_columns: Feature columns used during training
    """
    # Separate target from features
    X = cleaned_data.drop(columns=["ClaimCost"])
    y = cleaned_data["ClaimCost"]

    # Store feature columns for prediction-time alignment
    training_columns = X.columns.tolist()

    # Random Forest with bootstrap aggregation enabled
    model = RandomForestRegressor(
        n_estimators=100,
        bootstrap=True,
        random_state=42
    )

    model.fit(X, y)

    # Save trained model
    dump(model, "random_forest_model.joblib")

    return model, training_columns


def predict_claims(model, prediction_data, training_columns):
    """
    Generate claim cost predictions for new policyholders.

    Returns:
        DataFrame containing the original prediction data and
        PredictedClaimCost.
    """
    # Preserve original data
    original_data = prediction_data.copy()

    # Preprocess prediction data
    cleaned_prediction = preprocess_data(prediction_data)

    # Align prediction features with training features
    cleaned_prediction = cleaned_prediction.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Generate predictions
    predictions = model.predict(cleaned_prediction)

    # Append predictions to original data
    result = original_data.copy()
    result["PredictedClaimCost"] = predictions.round(2)

    # Save predictions
    result.to_csv(
        "predicted_claims.csv",
        index=False
    )

    return result


if __name__ == "__main__":
    # Load training and prediction datasets
    training_data = pd.read_csv("insurance_claims.csv")
    prediction_data = pd.read_csv("new_claims.csv")

    # Preprocess training data
    cleaned_data = preprocess_data(training_data)

    # Save cleaned training data
    cleaned_data.to_csv(
        "cleaned_insurance_data.csv",
        index=False
    )
    print("Cleaned data saved as 'cleaned_insurance_data.csv'")

    # Train model
    model, training_columns = train_model(cleaned_data)
    print("Model saved as 'random_forest_model.joblib'")

    # Generate predictions
    results = predict_claims(
        model,
        prediction_data,
        training_columns
    )

    print("Predictions saved to 'predicted_claims.csv'")
    print("\nSample Predictions:")
    print(results.head(10))