import pandas as pd
import numpy as np

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


NUMERICAL_COLUMNS = [
    "Age",
    "Annual_Income",
    "Last_Purchase_Amount",
    "Membership_Years",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "City",
    "Product_Category",
]

TARGET_COLUMN = "Purchase_Amount"


def preprocess_data(df):
    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN].copy()
        features = df.drop(columns=[TARGET_COLUMN])
    else:
        target = None
        features = df.copy()

    required_columns = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

    missing_columns = [
        col for col in required_columns
        if col not in features.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_COLUMNS
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS
        )
    ])

    features_processed = preprocessor.fit_transform(features)

    if hasattr(features_processed, "toarray"):
        features_processed = features_processed.toarray()

    return preprocessor, features_processed, target


def train_model(features_train, target_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(features_train, target_train)
    return model


def predict_data(model, preprocessor, df):
    features = df[
        NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    ].copy()

    features_processed = preprocessor.transform(features)

    predictions = model.predict(features_processed)
    predictions = np.round(predictions, 2)

    result_df = df.copy()
    result_df["Predicted_Purchase_Amount"] = predictions

    return result_df


if __name__ == "__main__":
    training_df = pd.read_csv("customer_data_raw.csv")

    preprocessor, features_processed, target = preprocess_data(
        training_df
    )

    model = train_model(
        features_processed,
        target
    )

    dump(model, "decision_tree_model.joblib")
    dump(preprocessor, "preprocessor.joblib")

    new_customers = pd.read_csv("new_customers.csv")

    predictions = predict_data(
        model,
        preprocessor,
        new_customers
    )

    predictions.to_csv(
        "predicted_customers.csv",
        index=False
    )

    print(predictions.head(10))


# DO NOT FORGET TO RUN THIS SCRIPT (python3 main.py)