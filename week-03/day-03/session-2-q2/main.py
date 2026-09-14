import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_dataset(file_path):
    """Loads the customer dataset from an Excel file."""
    return pd.read_excel(file_path)


def preprocess_data(df):
    """
    Cleans missing values, performs ordinal mapping on marital_status,
    and one-hot encodes non-ordinal categorical features (job, contact).
    """
    df = df.copy()

    # 1. Fill missing values
    # Numerical columns -> mean
    num_cols = ["age", "balance"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Categorical columns -> mode
    cat_cols = ["job", "contact", "marital_status"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 2. Ordinal Encoding for marital_status
    marital_mapping = {"single": 0, "married": 1, "divorced": 2}
    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].map(marital_mapping)

    # 3. One-Hot Encoding for job and contact with drop_first=True
    one_hot_cols = [c for c in ["job", "contact"] if c in df.columns]
    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)

    return df


def split_dataset(df):
    """Splits features (X) and target (y) into train and test sets."""
    X = df.drop(columns=["subscribed"])
    y = df["subscribed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_logistic_model(X_train, y_train):
    """Trains a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # 1. Load the dataset
    df = load_dataset("customer_data.xlsx")

    # 2. Print dataset preview
    print("Dataset Preview:")
    print(df.head())

    # 3. Preprocess the dataset
    processed_df = preprocess_data(df)

    # 4. Save processed dataset to disk
    processed_df.to_excel("processed_data.xlsx", index=False)

    # 5. Split data
    X_train, X_test, y_train, y_test = split_dataset(processed_df)

    # 6. Train Logistic Regression model
    model = train_logistic_model(X_train, y_train)

    # 7. Generate predictions
    predictions = model.predict(X_test)

    # 8. Print predictions output
    print("Predictions Output")
    print("Logistic Regression Predictions:")
    print(predictions)
