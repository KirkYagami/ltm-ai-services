import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


def clean_data(input_path: str, output_path: str) -> pd.DataFrame:

    df = pd.read_csv(input_path)

    # 1. Drop non-predictive booking columns EXCEPT order_id (kept for now).
    booking_cols = [
        "order_date",
        "center_code",
        "destination_city",
        "carrier_partner",
    ]
    df = df.drop(columns=booking_cols, errors="ignore")

    # 2. Normalise text BEFORE dropping duplicates.
    if "shipping_method" in df.columns:
        df["shipping_method"] = (
            df["shipping_method"].astype(str).str.strip().str.lower()
        )
        df["shipping_method"] = df["shipping_method"].map(
            {"standard": 0, "express": 1, "priority": 2}
        )

    if "is_gift_order" in df.columns:
        df["is_gift_order"] = (
            df["is_gift_order"].astype(str).str.strip().str.lower()
        )
        df["is_gift_order"] = df["is_gift_order"].map({"no": 0, "yes": 1})

    # 3. NOW drop duplicates. Exclude order_id from the dup check so that
    #    two rows differing only in order_id are still treated as duplicates.
    feature_cols = [c for c in df.columns if c != "order_id"]
    df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    # 4. Impute missing numerics with each column's median.
    for col in ["package_weight_kg", "item_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # 5. Coerce the outcome column if present (training only).
    if "is_returned" in df.columns:
        df["is_returned"] = pd.to_numeric(df["is_returned"], errors="coerce")

    # 6. Drop order_id so only predictive columns remain.
    df = df.drop(columns=["order_id"], errors="ignore")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def train_model(processed_path: str, model_path: str) -> RandomForestClassifier:
    df = pd.read_csv(processed_path)
    X = df.drop(columns=["is_returned"])
    y = df["is_returned"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)

    return model


def predict(
    model_path: str,
    input_path: str,
    processed_path: str,
    output_path: str,
) -> pd.DataFrame:
    raw = pd.read_csv(input_path)


    # --- Mirror clean_data's normalisation + dedup to derive order_ids ---
    tmp = raw.copy()

    if "shipping_method" in tmp.columns:
        tmp["shipping_method"] = (
            tmp["shipping_method"].astype(str).str.strip().str.lower()
        )
        tmp["shipping_method"] = tmp["shipping_method"].map(
            {"standard": 0, "express": 1, "priority": 2}
        )

    if "is_gift_order" in tmp.columns:
        tmp["is_gift_order"] = (
            tmp["is_gift_order"].astype(str).str.strip().str.lower()
        )
        tmp["is_gift_order"] = tmp["is_gift_order"].map({"no": 0, "yes": 1})

    feature_cols = [
        c for c in tmp.columns
        if c not in ("order_id", "order_date", "center_code",
                     "destination_city", "carrier_partner")
    ]
    tmp = tmp.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    order_ids = tmp["order_id"].reset_index(drop=True)

    # --- Clean the pending batch with the shared cleaning stage ---
    cleaned = clean_data(input_path, processed_path)

    assert len(order_ids) == len(cleaned), (
        f"Row mismatch: order_ids={len(order_ids)}, cleaned={len(cleaned)}"
    )

    # --- Load the stored classifier (no refitting) ---
    model = joblib.load(model_path)

    # --- Issue one indicator per order ---
    predictions = model.predict(cleaned)

    result = pd.DataFrame(
        {
            "order_id": order_ids,
            "predicted_is_returned": predictions,
        }
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)

    return result


def main():

    raw_data_path = "data/order_return_train.csv"
    processed_data_path = "processed_data/order_return_train_cleaned.csv"

    predict_raw = "data/order_return_predict.csv"
    predict_cleaned = "processed_data/order_return_predict_cleaned.csv"

    model_path = "artifacts/return_risk_model.pkl"

    final_predictions_path = "output/order_return_predictions.csv"

    train_cleaned = clean_data(raw_data_path, processed_data_path)
    
    model = train_model(processed_data_path, model_path)

    final_predictions_df = predict(model_path,
                                    predict_raw,
                                    predict_cleaned,
                                    final_predictions_path)

    # pd.DataFrame({"": })

    print("", pd.read_csv(processed_data_path).shape)
    print("", model.n_features_in_)
    print("", final_predictions_df.shape)


main()
