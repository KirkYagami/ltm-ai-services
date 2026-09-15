import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def preprocess_data():
    """Preprocess the training customer data."""

    df = pd.read_csv("customer_behavior.csv")

    target_column = "Purchased"

    numerical_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include="number")
        .columns
    )

    categorical_columns = (
        df.drop(columns=[target_column])
        .select_dtypes(include=["object", "category"])
        .columns
    )

    # Fill missing numerical values with column mean
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing categorical values with column mode
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Save cleaned data before encoding/scaling
    df.to_csv("cleaned_customer_data.csv", index=False)

    # Separate features and target
    features = df.drop(columns=[target_column])
    target = df[target_column]

    # One-hot encode categorical columns
    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dtype=int
    )

    # Scale numerical columns
    scaler = StandardScaler()
    features[numerical_columns] = scaler.fit_transform(
        features[numerical_columns]
    )

    # Save fitted scaler
    joblib.dump(scaler, "scaler.pkl")

    training_columns = features.columns.tolist()

    return features, target, scaler, training_columns


def train_model(features, target):
    """Train the KNN classifier."""

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(features, target)

    joblib.dump(model, "knn_model.pkl")

    return model


def predict_customers(model, scaler, training_columns):
    """Predict purchases for new customers."""

    new_customers = pd.read_csv("new_customers.csv")

    # Keep original data for final output
    result = new_customers.copy()

    # One-hot encode categorical columns
    categorical_columns = new_customers.select_dtypes(
        include=["object", "category"]
    ).columns

    encoded = pd.get_dummies(
        new_customers,
        columns=categorical_columns,
        dtype=int
    )

    # Align prediction columns with training columns
    encoded = encoded.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Identify original numerical columns
    numerical_columns = new_customers.select_dtypes(
        include="number"
    ).columns

    # Apply the fitted scaler
    encoded[numerical_columns] = scaler.transform(
        encoded[numerical_columns]
    )

    predictions = model.predict(encoded)

    result["Purchased"] = predictions

    result.to_csv("predicted_customers.csv", index=False)

    print("Predictions saved to predicted_customers.csv")
    print(result)

    return result


if __name__ == "__main__":
    features, target, scaler, training_columns = preprocess_data()

    model = train_model(features, target)

    predict_customers(
        model,
        scaler,
        training_columns
    )