import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


def load_and_process_data(train_df, test_df):
    """
    Cleans, encodes, and prepares both training and test datasets for Linear Regression modeling.
    """
    # 1. Feature Columns Selection & Missing Value Handling
    feature_cols = ["Bedrooms", "Location", "Area", "Garage"]
    target_col = "Price"

    # Drop missing values from both raw datasets
    train_df = train_df.dropna().copy()
    test_df = test_df.dropna().copy()

    # Select required columns
    X_train_raw = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test_raw = test_df[feature_cols].copy()

    # Store cleaned unencoded test dataframe
    original_test_df = X_test_raw.copy()

    # 2. Categorical Column Handling (Explicit String Conversion)
    categorical_cols = ["Location", "Garage"]
    for col in categorical_cols:
        X_train_raw[col] = X_train_raw[col].astype(str)
        X_test_raw[col] = X_test_raw[col].astype(str)

    # 3. Encoding Setup using ColumnTransformer & OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    # Fit encoder on training features and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)

    return X_train_processed, y_train, X_test_processed, original_test_df


def train_model(features, target):
    """
    Trains a Linear Regression model using processed training data.
    """
    model = LinearRegression()
    model.fit(features, target)
    return model


def predict(model, processed_test_data, original_test_data):
    """
    Generates house price predictions for new property listings using the trained model.
    """
    predictions = model.predict(processed_test_data)
    result_df = original_test_data.copy()
    result_df["Predicted Price"] = predictions
    return result_df


# -------------------------------------------------------------
# Top-level Execution Routine
# Runs on module import to guarantee output files are created
# -------------------------------------------------------------
try:
    train_df = pd.read_excel("house_data.xlsx")
    test_df = pd.read_excel("predictions.xlsx")

    # Data Preprocessing
    X_train, y_train, X_test, original_test_df = load_and_process_data(train_df, test_df)

    # Save Cleaned Training Data
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    cleaned_df = pd.DataFrame(X_train_dense)
    cleaned_df["Price"] = y_train.values
    cleaned_df.to_excel("cleaned_data.xlsx", index=False)

    # Model Training & Predictions
    model = train_model(X_train, y_train)
    result_df = predict(model, X_test, original_test_df)
    result_df.to_excel("predicted_prices.xlsx", index=False)

    # Console Display
    print("Top 10 Predicted Prices for Test Data:")
    display_cols = ["Bedrooms", "Location", "Area", "Predicted Price"]
    print(result_df[display_cols].head(10).to_string())
except FileNotFoundError:
    pass
