import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def preprocess_data(dataframe):
    """
    Preprocess the electricity dataset.

    - Fills numerical missing values with the median.
    - Fills categorical missing values with the mode.
    - One-hot encodes categorical columns using drop_first=True.

    Returns:
        pandas.DataFrame: Cleaned and encoded dataset.
    """
    df = dataframe.copy()

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

    Returns:
        model: Trained RandomForestRegressor.
        features_test: Test feature DataFrame.
        target_test: Test target Series.
        training_columns: List of feature column names.
    """
    # Separate features and target
    features = cleaned_data.drop(
        columns=["MonthlyElectricityBill"]
    )
    target = cleaned_data["MonthlyElectricityBill"]

    # Store feature columns used during training
    training_columns = features.columns.tolist()

    # 80/20 train-test split
    features_train, features_test, target_train, target_test = (
        train_test_split(
            features,
            target,
            test_size=0.20,
            random_state=42
        )
    )

    # Random Forest with bootstrap aggregation
    model = RandomForestRegressor(
        n_estimators=100,
        bootstrap=True,
        random_state=42
    )

    # Train model
    model.fit(features_train, target_train)

    # Save trained model
    dump(model, "random_forest_model.joblib")

    return (
        model,
        features_test,
        target_test,
        training_columns
    )


def evaluate_model(
    model,
    features_test,
    target_test,
    training_columns
):
    """
    Evaluate the Random Forest model and calculate feature importance.

    Returns:
        evaluation_metrics: Dictionary containing RMSE, MAE and R2.
        feature_importance: DataFrame sorted by importance descending.
    """
    # Generate predictions
    predictions = model.predict(features_test)

    # Calculate evaluation metrics
    rmse = mean_squared_error(
        target_test,
        predictions
    ) ** 0.5

    mae = mean_absolute_error(
        target_test,
        predictions
    )

    r2 = r2_score(
        target_test,
        predictions
    )

    evaluation_metrics = {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    }

    # Extract feature importance
    feature_importance = pd.DataFrame({
        "Feature": training_columns,
        "Importance": model.feature_importances_
    })

    # Sort by importance descending
    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    return evaluation_metrics, feature_importance


if __name__ == "__main__":
    # Load training data
    training_data = pd.read_csv("electricity_data.csv")

    # Preprocess data
    cleaned_data = preprocess_data(training_data)

    # Save cleaned dataset
    cleaned_data.to_csv(
        "cleaned_electricity_data.csv",
        index=False
    )

    print(
        "Cleaned data saved as 'cleaned_electricity_data.csv'"
    )

    # Train model
    (
        model,
        features_test,
        target_test,
        training_columns
    ) = train_model(cleaned_data)

    print(
        "Model saved as 'random_forest_model.joblib'"
    )

    # Evaluate model
    evaluation_metrics, feature_importance = evaluate_model(
        model,
        features_test,
        target_test,
        training_columns
    )

    print()
    print(f"RMSE: {evaluation_metrics['RMSE']:.2f}")
    print(f"MAE: {evaluation_metrics['MAE']:.2f}")
    print(f"R2 Score: {evaluation_metrics['R2']:.4f}")

    # Save feature importance
    feature_importance.to_csv(
        "feature_importance.csv",
        index=False
    )

    print()
    print("Feature Importance:")
    print(feature_importance)