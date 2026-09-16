import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def preprocess_data():
    """Preprocess customer data for clustering."""

    df = pd.read_csv("ecommerce_customers.csv")

    numerical_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Fill numerical missing values with mean
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill categorical missing values with mode
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Save cleaned dataset before encoding/scaling
    df.to_csv("cleaned_ecommerce_data.csv", index=False)

    # Keep cleaned original data for segment assignment
    original_dataset = df.copy()

    # One-hot encode categorical columns
    features = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False
    )

    # Standardize numerical columns
    scaler = StandardScaler()
    features[numerical_columns] = scaler.fit_transform(
        features[numerical_columns]
    )

    joblib.dump(scaler, "scaler.pkl")

    return features, original_dataset


def train_model(features):
    """Train and save the K-Means model."""

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    model.fit(features)

    joblib.dump(model, "kmeans_model.pkl")

    return model


def assign_segments(model, original_dataset):
    """Assign cluster labels to customers."""

    segmented_data = original_dataset.copy()

    segmented_data["CustomerSegment"] = model.labels_

    segmented_data.to_csv(
        "segmented_customers.csv",
        index=False
    )

    numerical_columns = original_dataset.select_dtypes(
        include="number"
    ).columns.tolist()

    summary = segmented_data.groupby(
        "CustomerSegment"
    )[numerical_columns].mean()

    print("\nCustomer Segmentation Summary:")
    print(summary.round(2))

    return segmented_data


if __name__ == "__main__":
    features, original_dataset = preprocess_data()

    model = train_model(features)

    assign_segments(model, original_dataset)