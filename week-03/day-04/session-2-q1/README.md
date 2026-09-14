# Insurance Claim Cost Prediction Using Random Forest

## Objective

Build a Random Forest Regression pipeline to predict insurance claim
costs for new policyholders.

The solution must:

- Clean and preprocess the insurance dataset.
- Handle missing numerical and categorical values.
- Encode categorical features.
- Train a Random Forest Regressor.
- Use bootstrap aggregation (bagging).
- Predict claim costs for new policyholders.
- Save the trained model and prediction results to disk.

---

## Project Structure

```text
Project/
├── insurance_claims.csv
├── new_claims.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_insurance_data.csv       [Generated]
├── random_forest_model.joblib       [Generated]
└── predicted_claims.csv             [Generated]
```

### Important Note

A `.gitignore` file may already exist in the workspace outside the
project directory. Do not remove or delete it.

---

## Dataset

### Training Dataset

File:

```text
insurance_claims.csv
```

The dataset contains historical policyholder records and the target
variable `ClaimCost`.

### Columns

| Column | Type | Description |
|---|---|---|
| PolicyID | Integer | Unique policy identifier |
| Age | Float | Age of the policyholder |
| Gender | String | Male / Female |
| VehicleType | String | Car / Bike / Truck |
| VehicleAge | Float | Age of the vehicle |
| AnnualMileage | Float | Annual distance driven |
| PolicyType | String | Comprehensive / Third-Party |
| ClaimHistory | Float | Number of previous claims |
| ClaimCost | Integer | Total insurance claim cost and target |

Approximately 3% missing values may be present across columns except
`PolicyID` and `ClaimCost`.

---

## Prediction Dataset

File:

```text
new_claims.csv
```

This dataset contains the same input columns as the training dataset
except for:

```text
ClaimCost
```

`ClaimCost` must be predicted by the trained model.

---

# Function 1: preprocess_data

## Signature

```python
preprocess_data(dataframe)
```

## Parameters

`dataframe`:

A pandas DataFrame containing either:

- the training dataset, or
- the prediction dataset.

## Required Logic

### 1. Remove PolicyID

`PolicyID` is only an identifier and must not be used as a model
feature.

Remove:

```text
PolicyID
```

### 2. Identify numerical columns

Identify numerical columns using pandas data types.

Numerical columns include fields such as:

```text
Age
VehicleAge
AnnualMileage
ClaimHistory
ClaimCost
```

when `ClaimCost` is present.

### 3. Handle missing numerical values

Fill missing numerical values using the median of their respective
columns.

Example:

```python
df[column] = df[column].fillna(df[column].median())
```

### 4. Identify categorical columns

Identify categorical columns using pandas object/string data types.

Expected categorical columns include:

```text
Gender
VehicleType
PolicyType
```

### 5. Handle missing categorical values

Fill missing categorical values using the mode of each column.

Example:

```python
df[column] = df[column].fillna(df[column].mode()[0])
```

### 6. One-hot encode categorical columns

Apply one-hot encoding to categorical columns using:

```python
pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
```

The resulting DataFrame must contain no object columns.

### Return

Return:

```text
pandas.DataFrame
```

containing the cleaned and encoded data.

---

# Function 2: train_model

## Signature

```python
train_model(cleaned_data)
```

## Parameters

`cleaned_data`:

The preprocessed training DataFrame returned by
`preprocess_data()`.

It contains:

```text
ClaimCost
```

as the target column.

## Required Logic

### 1. Separate target

Separate:

```text
ClaimCost
```

from the feature columns.

The target is:

```python
y = cleaned_data["ClaimCost"]
```

The features are all remaining columns.

### 2. Store training feature columns

Store the feature column names as a list.

These columns will be used later to align the prediction dataset with
the training dataset.

### 3. Create Random Forest Regressor

Initialize:

```python
RandomForestRegressor(
    n_estimators=100,
    bootstrap=True,
    random_state=42
)
```

The model must use bootstrap aggregation.

### 4. Train the model

Fit the Random Forest using all available training features and target
values.

### 5. Save the model

Save the trained model using joblib as:

```text
random_forest_model.joblib
```

### Return

Return:

```text
model
training_columns
```

where:

- `model` is the trained `RandomForestRegressor`.
- `training_columns` is the list of feature column names used during
  training.

---

# Function 3: predict_claims

## Signature

```python
predict_claims(model, prediction_data, training_columns)
```

## Parameters

### model

The trained `RandomForestRegressor` returned by `train_model()`.

### prediction_data

A pandas DataFrame containing new policyholder records loaded from:

```text
new_claims.csv
```

### training_columns

The feature column names returned by `train_model()`.

These must be used to make sure prediction features have exactly the
same structure as training features.

## Required Logic

### 1. Preserve original data

Create a copy of the original prediction DataFrame before
preprocessing.

The output must preserve all original columns.

### 2. Preprocess prediction data

Call:

```python
preprocess_data(prediction_data)
```

The same preprocessing logic used for training must be applied.

### 3. Align features

The encoded prediction DataFrame must be aligned with the training
feature columns.

Use:

```python
cleaned_prediction.reindex(
    columns=training_columns,
    fill_value=0
)
```

This ensures that:

- missing training columns are added with value `0`.
- extra columns are removed.
- column order matches training.
- the model receives the expected feature schema.

### 4. Generate predictions

Use:

```python
model.predict(cleaned_prediction)
```

### 5. Create output DataFrame

Preserve every original column from `new_claims.csv`.

Append:

```text
PredictedClaimCost
```

The prediction values must be rounded to 2 decimal places.

### 6. Save output

Save the resulting DataFrame as:

```text
predicted_claims.csv
```

### Return

Return:

```text
pandas.DataFrame
```

containing the original prediction data plus:

```text
PredictedClaimCost
```

---

# Main Execution Block

The main block must execute only when:

```python
if __name__ == "__main__":
```

The execution flow should be:

1. Load `insurance_claims.csv`.
2. Load `new_claims.csv`.
3. Call `preprocess_data()` on the training data.
4. Save the cleaned training data as:
   `cleaned_insurance_data.csv`.
5. Call `train_model()` using the cleaned training data.
6. Save the trained model as:
   `random_forest_model.joblib`.
7. Call `predict_claims()` using:
   - the trained model,
   - prediction data,
   - training columns.
8. Save predictions as:
   `predicted_claims.csv`.
9. Print status messages.
10. Print a sample of the predictions.

---

# Generated Files

Running:

```bash
python3 main.py
```

must generate:

```text
cleaned_insurance_data.csv
random_forest_model.joblib
predicted_claims.csv
```

### cleaned_insurance_data.csv

Contains the preprocessed and encoded training dataset.

### random_forest_model.joblib

Contains the trained Random Forest Regression model.

### predicted_claims.csv

Contains:

- all original columns from `new_claims.csv`
- `PredictedClaimCost`

The prediction column must contain numeric values and must not contain
null values.

---

# Expected Console Output

The console should contain messages similar to:

```text
Cleaned data saved as 'cleaned_insurance_data.csv'
Model saved as 'random_forest_model.joblib'
Predictions saved to 'predicted_claims.csv'

Sample Predictions:
```

The exact prediction values depend on the supplied dataset.

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

Run the complete test suite using:

```bash
python3 -m pytest tests.py -v
```

The implementation must satisfy tests for:

- Function availability.
- Missing-value handling.
- Removal of `PolicyID`.
- Categorical encoding.
- Random Forest model creation.
- Joblib model creation.
- Prediction file creation.
- `PredictedClaimCost` column creation.
- Prediction row-count preservation.

---

# Key Requirements

The implementation must:

- Use `pandas` for data processing.
- Use `RandomForestRegressor`.
- Use exactly 100 estimators.
- Enable bootstrap aggregation.
- Use `random_state=42`.
- Remove `PolicyID`.
- Fill numerical missing values with the median.
- Fill categorical missing values with the mode.
- Use one-hot encoding with `drop_first=True`.
- Produce a fully numeric training DataFrame.
- Save the model as `random_forest_model.joblib`.
- Preserve original prediction-data columns.
- Add `PredictedClaimCost`.
- Round predictions to 2 decimal places.
- Save predictions as `predicted_claims.csv`.
- Ensure prediction row count matches `new_claims.csv`.