# PROJECT_INSTRUCTIONS.md

# Customer Purchase Amount Prediction — Decision Tree Regression

## Objective

Build a machine learning regression pipeline to predict customer purchase
amounts using a `DecisionTreeRegressor`.

The project must:

- Load historical customer data from CSV.
- Handle missing numerical and categorical values.
- Encode categorical features.
- Preprocess data using `Pipeline` and `ColumnTransformer`.
- Train a `DecisionTreeRegressor`.
- Save the trained model using `joblib`.
- Save the fitted preprocessor using `joblib`.
- Load new customer data.
- Generate purchase amount predictions.
- Round predictions to two decimal places.
- Export predictions to a CSV file.

---

## Project Structure

```text
Project/
├── main.py
├── customer_data_raw.csv
├── new_customers.csv
├── tests.py
└── installation.txt
```

---

## Dataset

### Training Dataset

File:

```text
customer_data_raw.csv
```

Contains approximately 15,000 historical customer records.

The dataset contains the following columns:

| Column | Type | Description |
|---|---|---|
| Age | float64 | Customer age |
| Gender | object | Male / Female |
| City | object | Customer city |
| Annual_Income | float64 | Annual income |
| Last_Purchase_Amount | float64 | Amount spent on last purchase |
| Membership_Years | float64 | Number of years as a member |
| Product_Category | object | Product category |
| Purchase_Amount | float64 | Target variable |

The training dataset may contain missing values in feature columns.

`Purchase_Amount` is the target variable and does not contain missing
values.

### Prediction Dataset

File:

```text
new_customers.csv
```

Contains new customer records without the `Purchase_Amount` target column.

It has the same feature columns as the training dataset.

The prediction dataset may contain missing values.

---

# Required Functions

## Task 1 — Preprocess Data

Implement:

```python
preprocess_data(df)
```

### Requirements

If `Purchase_Amount` exists:

- Separate it from the feature columns.
- Return it as the target.

If `Purchase_Amount` does not exist:

- Set the target to `None`.

Numerical columns:

```text
Age
Annual_Income
Last_Purchase_Amount
Membership_Years
```

Categorical columns:

```text
Gender
City
Product_Category
```

Create a numerical pipeline using:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])
```

Create a categorical pipeline using:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

Combine both pipelines using:

```python
ColumnTransformer
```

Fit and transform the features using the `ColumnTransformer`.

The returned processed features must contain no missing values.

### Return

The function must return:

```text
preprocessor
features_processed
target
```

Where:

- `preprocessor` is the fitted `ColumnTransformer`.
- `features_processed` is the transformed feature matrix.
- `target` is the `Purchase_Amount` Series or `None`.

---

# Task 2 — Train Model

Implement:

```python
train_model(features_train, target_train)
```

### Requirements

Use:

```python
DecisionTreeRegressor(random_state=42)
```

Fit the model using:

```text
features_train
target_train
```

### Return

Return the trained `DecisionTreeRegressor`.

---

# Task 3 — Predict Data

Implement:

```python
predict_data(model, preprocessor, df)
```

### Requirements

Use the following feature columns:

```text
Age
Gender
City
Annual_Income
Last_Purchase_Amount
Membership_Years
Product_Category
```

Transform the new customer data using the already fitted preprocessor.

Use:

```python
preprocessor.transform(...)
```

Do not use:

```python
preprocessor.fit_transform(...)
```

for prediction data.

Generate predictions using the trained Decision Tree model.

Round predictions to two decimal places.

Create a copy of the input DataFrame and add:

```text
Predicted_Purchase_Amount
```

### Return

Return the resulting DataFrame containing:

- All original customer columns.
- `Predicted_Purchase_Amount`.

---

# Model Persistence

Save the trained model as:

```text
decision_tree_model.joblib
```

Save the fitted preprocessor as:

```text
preprocessor.joblib
```

Use `joblib.dump()`.

The saved model and preprocessor must be usable for future predictions.

---

# Main Execution

The main execution block must use:

```python
if __name__ == "__main__":
```

The execution flow should be:

1. Load `customer_data_raw.csv`.
2. Call `preprocess_data()`.
3. Train the Decision Tree model using `train_model()`.
4. Save the model to:
   `decision_tree_model.joblib`
5. Save the preprocessor to:
   `preprocessor.joblib`
6. Load `new_customers.csv`.
7. Generate predictions using `predict_data()`.
8. Save predictions to:
   `predicted_customers.csv`
9. Print the first 10 prediction rows.

---

# Required Output Files

After running:

```bash
python3 main.py
```

the following files must exist:

```text
decision_tree_model.joblib
preprocessor.joblib
predicted_customers.csv
```

The prediction file must contain:

```text
Predicted_Purchase_Amount
```

and must have the same number of rows as:

```text
new_customers.csv
```

All prediction values must be non-null.

---

# Important Preprocessing Rules

All missing feature values must be handled.

Numerical missing values:

```python
SimpleImputer(strategy="mean")
```

Categorical missing values:

```python
SimpleImputer(strategy="most_frequent")
```

Categorical features must be encoded using:

```python
OneHotEncoder(handle_unknown="ignore")
```

Use:

```python
ColumnTransformer
```

to apply the correct preprocessing to each feature type.

Do not manually encode categorical columns.

Do not fit the preprocessor again on prediction data.

---

# Model Requirements

The model must be:

```python
DecisionTreeRegressor(random_state=42)
```

This is a regression problem.

Do not use:

```text
LogisticRegression
DecisionTreeClassifier
RandomForestClassifier
```

or other classification models.

---

# Prediction Requirements

Predictions must be numerical purchase amounts.

Example:

```text
2409.58
2933.76
3196.91
2156.01
```

Predictions must be rounded to two decimal places.

The output column must be named exactly:

```text
Predicted_Purchase_Amount
```

---

# Testing

Install dependencies:

```bash
pip install -r installation.txt
```

Run the main program first:

```bash
python3 main.py
```

Then run the tests:

```bash
python3 -m pytest tests.py -v
```

The tests verify:

- Required functions exist.
- Preprocessing works.
- Missing values are handled.
- Target extraction works.
- Prediction data without a target is supported.
- Model file is created.
- Preprocessor file is created.
- Prediction CSV is created.
- Prediction column exists.
- Predictions contain no null values.
- Prediction row count matches the input dataset.

---

# Expected Execution Flow

```text
customer_data_raw.csv
        |
        v
preprocess_data()
        |
        +--> fitted ColumnTransformer
        |
        +--> processed features
        |
        +--> Purchase_Amount target
        |
        v
train_model()
        |
        v
DecisionTreeRegressor
        |
        +--------------------+
        |                    |
        v                    v
decision_tree_model.joblib   preprocessor.joblib
                             |
                             v
                     new_customers.csv
                             |
                             v
                    preprocessor.transform()
                             |
                             v
                       model.predict()
                             |
                             v
                Predicted_Purchase_Amount
                             |
                             v
                  predicted_customers.csv
```

# Important Notes

- The preprocessing pipeline must be fitted using training data.
- The fitted preprocessor must be reused for new customer data.
- Do not call `fit()` or `fit_transform()` on the prediction dataset.
- Preserve all original columns in the prediction output.
- Add only the required `Predicted_Purchase_Amount` column.
- Ensure the generated files exist before running the tests.