import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


DATASET_PATH = "patient_readmission.csv"
CLEANED_PATH = "cleaned_patient_data.csv"
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "naive_bayes_model.pkl"
TARGET_COLUMN = "Readmitted"


def preprocess_data():
    """Load, clean, encode, scale and split the patient readmission dataset."""
    df = pd.read_csv(DATASET_PATH)

    # Identify numerical and categorical columns automatically.
    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # The target is numeric but must not be imputed or scaled.
    if TARGET_COLUMN in numerical_columns:
        numerical_columns.remove(TARGET_COLUMN)

    # Fill missing numerical values with the column mean.
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    # Fill missing categorical values with the column mode.
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # Save the cleaned dataset.
    df.to_csv(CLEANED_PATH, index=False)

    # Separate the target from the features.
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # One-hot encode the categorical columns.
    X = pd.get_dummies(X, columns=categorical_columns)

    # Scale the numerical columns with a fitted StandardScaler.
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Persist the fitted scaler.
    joblib.dump(scaler, SCALER_PATH)

    # 80/20 train-test split with stratification on the target.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Fit a Gaussian Naive Bayes classifier and persist it."""
    model = GaussianNB()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)

    return model


def evaluate_model(model, X_test, y_test):
    """Score the trained model and report accuracy and F1 score."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, f1


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)