import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_dataset(file_path):
    """
    Load a CSV dataset into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def preprocess_data(df, encoder=None, fit_encoder=True):
    """
    Handle missing values and encode EducationLevel.

    YearsExperience:
        Missing values are replaced with the column mean.

    EducationLevel:
        Missing values are replaced with the mode.

    LabelEncoder:
        A new encoder is fitted during training.
        The existing encoder is reused during prediction.
    """

    # Work on a copy so the original DataFrame is not modified
    df = df.copy()

    # Handle missing YearsExperience using mean
    if df["YearsExperience"].isnull().any():
        df["YearsExperience"] = df["YearsExperience"].fillna(
            df["YearsExperience"].mean()
        )

    # Handle missing EducationLevel using mode
    if df["EducationLevel"].isnull().any():
        df["EducationLevel"] = df["EducationLevel"].fillna(
            df["EducationLevel"].mode()[0]
        )

    # Create and fit encoder during training
    if encoder is None:
        encoder = LabelEncoder()

    if fit_encoder:
        df["EducationLevel"] = encoder.fit_transform(
            df["EducationLevel"]
        )
    else:
        # Reuse the encoder during prediction
        df["EducationLevel"] = encoder.transform(
            df["EducationLevel"]
        )

    return df, encoder


def split_data(df):
    """
    Separate features and target and perform train-test split.

    Features:
        YearsExperience
        EducationLevel

    Target:
        Salary
    """

    X = df[["YearsExperience", "EducationLevel"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Linear Regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def predict_new_data(model, encoder, input_file, output_file):
    """
    Load new candidate data, preprocess it using the existing encoder,
    generate salary predictions, and save the results.
    """

    # Load prediction dataset
    prediction_df = load_dataset(input_file)

    # Apply the same preprocessing rules.
    # fit_encoder=False is important because we must NOT
    # create a new category mapping during prediction.
    processed_df, _ = preprocess_data(
        prediction_df,
        encoder=encoder,
        fit_encoder=False
    )

    # Select the same features used during training
    X_prediction = processed_df[
        ["YearsExperience", "EducationLevel"]
    ]

    # Generate predictions
    predictions = model.predict(X_prediction)

    # Create output DataFrame
    result = processed_df[
        ["YearsExperience", "EducationLevel"]
    ].copy()

    result["PredictedSalary"] = predictions

    # Save predictions
    result.to_csv(output_file, index=False)

    # Required console output
    print("Predicted Salaries:")
    print(result)

    return result


if __name__ == "__main__":

    # File paths
    training_file = "salary_data.csv"
    prediction_file = "predictions.csv"
    output_file = "predicted_salaries.csv"

    # 1. Load training data
    df = load_dataset(training_file)

    # 2. Preprocess training data
    processed_df, encoder = preprocess_data(df)

    # 3. Split data
    X_train, X_test, y_train, y_test = split_data(processed_df)

    # 4. Train model
    model = train_model(X_train, y_train)

    # 5. Predict new candidates
    predict_new_data(
        model,
        encoder,
        prediction_file,
        output_file
    )