# Customer Churn Prediction – Support Vector Classification

## Project Overview

This project builds a Support Vector Classification (SVC) machine learning pipeline to predict whether a customer will churn.

The pipeline:

- Cleans and preprocesses customer data.
- Handles missing numerical and categorical values.
- Applies one-hot encoding to categorical features.
- Trains an SVC model using a linear kernel.
- Generates churn predictions for new customers.
- Saves the trained model and prediction results to disk.

## Project Structure

```text
Project/
├── churn_data.csv
├── new_customers.csv
├── main.py
├── tests.py
├── installation.txt
└── .gitignore
```

## Input Files

### `churn_data.csv`

Training dataset containing historical customer records and the `Churned` target column.

Columns:

- `CustomerType`
- `ContractType`
- `MonthlyCharges`
- `TenureMonths`
- `SupportCalls`
- `Churned`

`Churned` is the target variable:

- `0` = Retained
- `1` = Churned

### `new_customers.csv`

Prediction dataset containing new customer records.

It contains the same feature columns as `churn_data.csv`, except for the `Churned` target column.

## Required Functions

### 1. `preprocess_data(dataframe)`

Preprocesses the input DataFrame.

The function must:

1. Identify numerical and categorical columns.
2. Fill missing numerical values using the median of each column.
3. Fill missing categorical values using the most frequent value (mode).
4. Apply one-hot encoding to categorical columns.
5. Use `drop_first=True` during one-hot encoding.
6. Return the cleaned and encoded DataFrame.

The function should work for both training and prediction datasets.

### 2. `train_model(cleaned_data)`

Trains the SVC classification model.

The function must:

1. Separate `Churned` from the feature columns.
2. Store the feature column names in `training_columns`.
3. Create an SVC model with:
   - `kernel="linear"`
   - `random_state=42`
4. Train the model using the feature and target data.
5. Save the trained model as:

```text
svc_model.joblib
```

6. Return:

```text
model, training_columns
```

### 3. `predict_churn(model, prediction_data, training_columns)`

Generates churn predictions for new customers.

The function must:

1. Preserve the original prediction DataFrame.
2. Preprocess the prediction data using `preprocess_data()`.
3. Align prediction features with `training_columns`.
4. Add missing training columns with a default value of `0`.
5. Reorder columns to match the training feature order.
6. Generate predictions using the trained SVC model.
7. Add a `PredictedChurn` column to the original prediction data.
8. Save the result as:

```text
predicted_customers.csv
```

9. Return the resulting DataFrame.

## Main Execution

The main execution block should run only when `main.py` is executed directly:

```python
if __name__ == "__main__":
```

It should:

1. Load `churn_data.csv`.
2. Load `new_customers.csv`.
3. Preprocess the training data.
4. Save the cleaned training data as:

```text
cleaned_churn_data.csv
```

5. Train the SVC model.
6. Save the model as:

```text
svc_model.joblib
```

7. Generate predictions for `new_customers.csv`.
8. Save predictions as:

```text
predicted_customers.csv
```

9. Print status messages and a sample of the predictions.

## Generated Files

After successful execution, the following files should be created:

| File | Description |
|---|---|
| `cleaned_churn_data.csv` | Preprocessed training data |
| `svc_model.joblib` | Trained SVC model |
| `predicted_customers.csv` | New customer records with predicted churn labels |

## Handling Missing Values

The training dataset contains approximately 3% missing values in feature columns.

Missing values must be handled as follows.

### Numerical columns

Use the column median:

```python
df[column] = df[column].fillna(df[column].median())
```

### Categorical columns

Use the most frequent category:

```python
df[column] = df[column].fillna(df[column].mode().iloc[0])
```

The target column `Churned` does not contain missing values.

## Categorical Encoding

Categorical columns must be converted into numerical features using one-hot encoding:

```python
pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
```

This ensures that the SVC model receives numerical input.

## Feature Alignment

The categorical values in the prediction dataset may not produce exactly the same encoded columns as the training dataset.

Therefore, prediction features must be aligned with the training columns:

```python
cleaned_prediction = cleaned_prediction.reindex(
    columns=training_columns,
    fill_value=0
)
```

This ensures:

- The same features are supplied to the model.
- Missing encoded categories are represented by `0`.
- Feature ordering matches the training data.

## Running the Project

Navigate to the project directory:

```bash
cd Project
```

Install dependencies:

```bash
pip install -r installation.txt
```

Run the application:

```bash
python main.py
```

## Running Tests

Run the test suite using:

```bash
pytest tests.py
```

or:

```bash
python -m pytest tests.py
```

## Expected Test Coverage

The tests validate:

- `preprocess_data()` exists and is callable.
- Missing values are removed.
- No object columns remain after preprocessing.
- The `Churned` target remains available.
- `train_model()` exists and is callable.
- The SVC model file is created.
- Training feature columns are returned.
- `predict_churn()` exists and is callable.
- Prediction output file is created.
- `PredictedChurn` is present in the result.
- Prediction row count matches the input prediction dataset.

## Expected Output

Example console output:

```text
Cleaned data saved as 'cleaned_churn_data.csv'
Model saved as 'svc_model.joblib'
Predictions saved to 'predicted_customers.csv'

Sample Predictions:
  CustomerType ContractType  MonthlyCharges  TenureMonths  SupportCalls  PredictedChurn
0      Premium     TwoYear            20.66          30.1           1.1               0
1      Premium      Annual           109.86          46.6           6.1               1
2      Regular     TwoYear            75.22          70.3           2.4               0
```

## Important Notes

- Do not remove the `.gitignore` file created outside the project directory.
- The trained model expects the same feature structure used during training.
- `training_columns` must be retained and used to align prediction features.
- Do not include the `Churned` column when generating predictions.
- The original prediction data should be preserved when adding `PredictedChurn`.