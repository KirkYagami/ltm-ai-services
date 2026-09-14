# Electricity Bill Forecast - Random Forest Evaluation and Feature Importance

## Problem Statement

Build a Random Forest Regression model to predict monthly household
electricity bills from household characteristics and electricity usage
patterns.

The solution must preprocess the raw data, train a Random Forest
Regression model, evaluate its performance, and identify the most
important features contributing to electricity bill predictions.

---

## Objective

The project must:

- Clean and encode the electricity dataset.
- Handle missing numerical and categorical values.
- Train a Random Forest Regression model.
- Use bootstrap aggregation (bagging).
- Evaluate the model using RMSE, MAE, and R2 Score.
- Extract and rank feature importance.
- Save the cleaned dataset, trained model, and feature importance
  results.

---

## Project Structure

```text
Project/
├── electricity_data.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_electricity_data.csv       [Generated]
├── random_forest_model.joblib         [Generated]
└── feature_importance.csv             [Generated]
```

### Important Note

A `.gitignore` file may already exist in the workspace outside the
project directory. Do not remove or delete it.

---

# Dataset

## Training Dataset

File:

```text
electricity_data.csv
```

The dataset contains household electricity information and the target
variable:

```text
MonthlyElectricityBill
```

## Schema

| Column | Type | Description |
|---|---|---|
| HouseholdType | String | Type of household: Apartment, House, or Condo |
| NumberOfRooms | Float | Number of rooms in the household |
| ACUnits | Float | Number of air conditioning units |
| AverageDailyUsageHours | Float | Average daily electricity usage in hours |
| Region | String | Geographic region: North, South, East, or West |
| MonthlyElectricityBill | Float | Monthly electricity bill in USD and target variable |

Approximately 3% missing values may exist across the input columns.
Missing values must be handled during preprocessing.

---

# Function 1: preprocess_data

## Signature

```python
preprocess_data(dataframe)
```

## Parameter

`dataframe`:

A pandas DataFrame containing the raw electricity dataset.

## Required Logic

### 1. Identify numerical columns

Identify numerical columns using pandas data types.

### 2. Handle numerical missing values

Fill missing numerical values using the median of each column.

Example:

```python
df[column] = df[column].fillna(df[column].median())
```

### 3. Identify categorical columns

Identify categorical columns using object/string data types.

Expected categorical columns include:

```text
HouseholdType
Region
```

### 4. Handle categorical missing values

Fill missing categorical values using the most frequent value
(mode) of each column.

Example:

```python
df[column] = df[column].fillna(df[column].mode()[0])
```

### 5. One-hot encoding

Apply one-hot encoding to categorical columns with:

```python
pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
```

The resulting DataFrame must contain no object columns.

### Return

Return the cleaned and encoded:

```text
pandas.DataFrame
```

The returned DataFrame must contain:

```text
MonthlyElectricityBill
```

as the target column.

---

# Function 2: train_model

## Signature

```python
train_model(cleaned_data)
```

## Parameter

`cleaned_data`:

The preprocessed DataFrame returned by `preprocess_data()`.

## Required Logic

### 1. Separate target

The target column is:

```text
MonthlyElectricityBill
```

Separate it from the feature columns.

### 2. Store training columns

Store the feature column names in a list.

These columns are required for evaluation and feature importance.

### 3. Split the data

Split the dataset into:

```text
80% training
20% testing
```

Use:

```python
train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)
```

### 4. Initialize Random Forest

Create a:

```python
RandomForestRegressor(
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
```

The model must use bootstrap aggregation.

### 5. Train the model

Fit the model using the training feature and target data.

### 6. Save the model

Save the trained model using joblib as:

```text
random_forest_model.joblib
```

### Return

Return four values:

```text
model
features_test
target_test
training_columns
```

Where:

- `model` is the trained Random Forest model.
- `features_test` contains the test features.
- `target_test` contains the test target values.
- `training_columns` contains the feature names used during training.

---

# Function 3: evaluate_model

## Signature

```python
evaluate_model(
    model,
    features_test,
    target_test,
    training_columns
)
```

## Parameters

### model

The trained `RandomForestRegressor`.

### features_test

Test feature data returned by `train_model()`.

### target_test

Actual test target values returned by `train_model()`.

### training_columns

List of feature names used during model training.

## Required Logic

### 1. Generate predictions

Use the trained model to predict values for `features_test`.

```python
predictions = model.predict(features_test)
```

### 2. Calculate RMSE

Calculate Root Mean Squared Error.

The value must be rounded to 2 decimal places.

### 3. Calculate MAE

Calculate Mean Absolute Error.

The value must be rounded to 2 decimal places.

### 4. Calculate R2 Score

Calculate the coefficient of determination.

The value must be rounded to 4 decimal places.

### 5. Store evaluation metrics

Return the metrics in a dictionary containing:

```text
RMSE
MAE
R2
```

Example:

```python
{
    "RMSE": 17.22,
    "MAE": 13.18,
    "R2": 0.8609
}
```

The exact values depend on the supplied dataset.

### 6. Calculate feature importance

Use:

```python
model.feature_importances_
```

Create a DataFrame with:

```text
Feature
Importance
```

The `Feature` column must contain the names from
`training_columns`.

### 7. Sort feature importance

Sort the DataFrame by:

```text
Importance
```

in descending order.

### Return

Return:

```text
evaluation_metrics
feature_importance
```

where:

- `evaluation_metrics` is a dictionary.
- `feature_importance` is a pandas DataFrame.

The number of rows in `feature_importance` must equal the number of
training columns.

---

# Main Execution Block

The main execution block must use:

```python
if __name__ == "__main__":
```

The execution flow must be:

1. Load `electricity_data.csv`.
2. Call `preprocess_data()`.
3. Save the cleaned data as:
   `cleaned_electricity_data.csv`.
4. Call `train_model()`.
5. Save the trained model as:
   `random_forest_model.joblib`.
6. Call `evaluate_model()`.
7. Print RMSE.
8. Print MAE.
9. Print R2 Score.
10. Save feature importance as:
    `feature_importance.csv`.
11. Print the feature importance DataFrame.

---

# Generated Files

Running:

```bash
python3 main.py
```

must generate:

```text
cleaned_electricity_data.csv
random_forest_model.joblib
feature_importance.csv
```

### cleaned_electricity_data.csv

Contains the cleaned and one-hot encoded dataset.

### random_forest_model.joblib

Contains the trained Random Forest Regression model.

### feature_importance.csv

Contains the feature importance ranking with:

```text
Feature
Importance
```

sorted from highest to lowest importance.

---

# Expected Console Output

The output should contain messages similar to:

```text
Cleaned data saved as 'cleaned_electricity_data.csv'
Model saved as 'random_forest_model.joblib'

RMSE: 17.22
MAE: 13.18
R2 Score: 0.8609

Feature Importance:
```

The exact metric and feature importance values depend on the supplied
dataset.

---

# Installation

Install the required dependencies using:

```bash
pip install -r installation.txt
```

---

# Run the Program

From the project directory:

```bash
python3 main.py
```

---

# Run Tests

Run the test suite using:

```bash
python3 -m pytest tests.py -v
```

---

# Required Test Behavior

The implementation must satisfy tests verifying:

- `preprocess_data` exists and is callable.
- Preprocessing removes all missing values.
- Preprocessing produces no object columns.
- `MonthlyElectricityBill` remains present as the target.
- `train_model` exists and is callable.
- The Random Forest model is saved as
  `random_forest_model.joblib`.
- `train_model` returns test features and test targets of equal length.
- Training columns are returned.
- `evaluate_model` exists and is callable.
- Evaluation metrics are returned as a dictionary.
- Feature importance is returned as a pandas DataFrame.
- Feature importance contains one row per training feature.

---

# Model Configuration

The required Random Forest configuration is:

```python
RandomForestRegressor(
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
```

The required train/test split is:

```python
train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42
)
```

---

# Key Requirements

The final implementation must:

- Use pandas for data processing.
- Use `RandomForestRegressor`.
- Use 100 estimators.
- Enable bootstrap aggregation.
- Use `random_state=42`.
- Use an 80/20 train-test split.
- Fill numerical missing values with column medians.
- Fill categorical missing values with column modes.
- Use one-hot encoding with `drop_first=True`.
- Produce no object columns after preprocessing.
- Preserve `MonthlyElectricityBill` as the target.
- Save the trained model as `random_forest_model.joblib`.
- Calculate RMSE.
- Calculate MAE.
- Calculate R2 Score.
- Round RMSE and MAE to 2 decimal places.
- Round R2 to 4 decimal places.
- Calculate feature importance using
  `model.feature_importances_`.
- Sort feature importance in descending order.
- Save feature importance as `feature_importance.csv`.