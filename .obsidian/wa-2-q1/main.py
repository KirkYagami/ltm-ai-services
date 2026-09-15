import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLUMNS = [
    "distance_km",
    "package_weight_kg",
    "warehouse_load_percent",
    "courier_rating",
    "prior_delay_count",
    "shipping_speed",
    "is_weekend_dispatch",
]

METADATA_COLUMNS = [
    "order_id",
    "dispatch_date",
    "origin_warehouse_code",
    "destination_city",
    "courier_partner",
]

TARGET_COLUMN = "is_delayed"

SPEED_MAP = {"standard": 0, "express": 1, "priority": 2}
WEEKEND_MAP = {"no": 0, "yes": 1}


def clean_data(input_path, output_path):
    """Prepare a raw shipment register into a model-ready table."""
    df = pd.read_csv(input_path)

    # Remove duplicate records before any summary statistic is drawn.
    df = df.drop_duplicates().reset_index(drop=True)

    # Repair unreported readings using the column's own median.
    for column in ["distance_km", "package_weight_kg"]:
        if column in df.columns:
            df[column] = df[column].fillna(df[column].median())

    # Normalise text categories to a single lowercase, trimmed form, then encode.
    if "shipping_speed" in df.columns:
        df["shipping_speed"] = (
            df["shipping_speed"].astype(str).str.strip().str.lower().map(SPEED_MAP)
        )

    if "is_weekend_dispatch" in df.columns:
        df["is_weekend_dispatch"] = (
            df["is_weekend_dispatch"].astype(str).str.strip().str.lower().map(WEEKEND_MAP)
        )

    # Drop the five booking columns (metadata) and any other non-predictive text.
    drop_columns = [column for column in METADATA_COLUMNS if column in df.columns]
    df = df.drop(columns=drop_columns)

    # Keep only predictive columns plus the target if it is present.
    keep_columns = [column for column in FEATURE_COLUMNS if column in df.columns]
    if TARGET_COLUMN in df.columns:
        keep_columns = keep_columns + [TARGET_COLUMN]
    df = df[keep_columns]

    # Ensure output directory exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def train_model(processed_path, model_path):
    """Fit a fixed random forest classifier and persist it."""
    df = pd.read_csv(processed_path)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model


def predict_new_data(model_path, input_path, processed_path, output_path):
    """Load the stored classifier and score the pending batch."""
    # Hold order_id references from the raw batch.
    raw = pd.read_csv(input_path)
    order_ids = raw["order_id"]

    # Prepare the batch with the same cleaning stage used for training.
    cleaned = clean_data(input_path, processed_path)

    # Load the stored classifier, do not refit.
    model = joblib.load(model_path)

    predictions = model.predict(cleaned[FEATURE_COLUMNS])

    result = pd.DataFrame(
        {
            "order_id": order_ids.values,
            "predicted_is_delayed": predictions,
        }
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def main():
    train_raw_path = "data/shipment_delivery_train.csv"
    predict_raw_path = "data/shipment_delivery_predict.csv"

    train_processed_path = "processed_data/shipment_delivery_train_cleaned.csv"
    predict_processed_path = "processed_data/shipment_delivery_predict_cleaned.csv"

    model_path = "artifacts/delivery_delay_model.pkl"
    output_path = "output/shipment_delivery_predictions.csv"

    # Stage 1: clean the training register.
    train_cleaned = clean_data(train_raw_path, train_processed_path)
    print(f"Model-ready training register dimensions: {train_cleaned.shape}")

    # Stage 2: train the classifier.
    model = train_model(train_processed_path, model_path)
    print(f"Number of inputs the classifier was fitted on: {model.n_features_in_}")

    # Stage 3: score the pending batch.
    predictions = predict_new_data(
        model_path,
        predict_raw_path,
        predict_processed_path,
        output_path,
    )
    print(f"Issued indicator dimensions: {predictions.shape}")


if __name__ == "__main__":
    main()