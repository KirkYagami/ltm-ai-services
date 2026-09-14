# Car Purchase Prediction Using Decision Tree Classification

## Objective

Build a machine learning classification model that predicts whether a
customer is likely to purchase a car.

The project uses a `DecisionTreeClassifier` with the Gini impurity criterion.

## Project Structure

```text
Project/
├── car_purchase_data.csv
├── new_customers.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_car_purchase_data.csv
├── decision_tree_model.joblib
└── predicted_customers.csv
```

## Dataset

### Training Dataset

File:

```text
car_purchase_data.csv
```

Columns:

- Age
- Annual_Income
- Gender
- Marital_Status
- Occupation
- Education_Level
- Purchased

`Purchased` is the binary target.

Target encoding:

```text
Yes -> 1
No  -> 0
```

### Prediction Dataset

File:

```text
new_customers.csv
```

Contains the same feature columns as the training dataset but does not
contain the `Purchased` column.

---

## Preprocessing

The following numerical columns must be handled:

```text
Age
Annual_Income
```

Missing numerical values must be replaced using the median of the
corresponding column.

Categorical columns:

```text
Gender
Marital_Status
Occupation
Education_Level
```

Categorical features must be one-hot encoded using:

```python
pd.get_dummies(..., drop_first=True)
```

The target must be removed from the feature set after it is encoded.

The cleaned and encoded training data must be saved as:

```text
cleaned_car_purchase_data.csv
```

The function must be:

```python
preprocess_data(filepath)
```

and return:

```text
features
target
```

Both must be pandas objects.

---

## Model Training

Implement:

```python
train_model(features, target)
```

Split the data into:

```text
80% training
20% testing
```

using:

```python
train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)
```

Use:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
```

Train the classifier on the training data.

Evaluate the model using accuracy.

Accuracy must be rounded to four decimal places.

Calculate feature importance using:

```python
classifier.feature_importances_
```

Return the three most important feature names.

The function must return:

```text
classifier
accuracy
top_features
```

The trained model must be saved using joblib as:

```text
decision_tree_model.joblib
```

---

## Prediction

Implement:

```python
predict_customers(
    classifier,
    feature_columns,
    filepath
)
```

Load:

```text
new_customers.csv
```

Handle missing numerical values.

Apply the same categorical encoding used during training:

```python
pd.get_dummies(..., drop_first=True)
```

Align the resulting columns with the training feature columns.

Use:

```python
reindex(
    columns=feature_columns,
    fill_value=0
)
```

Generate predictions using the trained classifier.

Convert predictions back to:

```text
1 -> Yes
0 -> No
```

Add a new column:

```text
Predicted_Purchase
```

Save the resulting DataFrame as:

```text
predicted_customers.csv
```

Return the resulting DataFrame.

---

## Required Output Files

After executing:

```bash
python3 main.py
```

the following files must exist:

```text
cleaned_car_purchase_data.csv
decision_tree_model.joblib
predicted_customers.csv
```

The prediction file must contain:

```text
Predicted_Purchase
```

All prediction values must be non-null.

The prediction output must contain the same number of rows as
`new_customers.csv`.

---

## Main Execution

The main program must:

1. Load `car_purchase_data.csv`.
2. Preprocess the training data.
3. Train the Decision Tree classifier.
4. Calculate model accuracy.
5. Identify the top three features.
6. Save the trained model.
7. Load `new_customers.csv`.
8. Generate customer purchase predictions.
9. Save predictions to `predicted_customers.csv`.
10. Print the model accuracy.
11. Print the top three features.
12. Print the prediction results.

---

## Running the Project

Install dependencies:

```bash
pip install -r installation.txt
```

Run the program:

```bash
python3 main.py
```

Run the tests:

```bash
python3 -m pytest tests.py -v
```

The program should be executed once before running tests because several
tests verify that the generated output files already exist.

---

## Expected Output

The console should contain output similar to:

```text
Model Accuracy: 0.7534
Top 3 Features: ['Annual_Income', 'Age', 'Occupation_Clerk']

Predictions saved to predicted_customers.csv
```

The exact accuracy and feature rankings depend on the supplied dataset.

## Important Requirements

- Use `DecisionTreeClassifier`.
- Use `criterion="gini"`.
- Use `max_depth=5`.
- Use `random_state=42`.
- Use an 80/20 train-test split.
- Handle missing `Age` and `Annual_Income` values using the median.
- Use one-hot encoding with `drop_first=True`.
- Encode `Purchased` as Yes = 1 and No = 0.
- Save the cleaned training data.
- Save the trained model using joblib.
- Save predictions to `predicted_customers.csv`.
- Prediction column must be exactly `Predicted_Purchase`.
- Prediction values must be `Yes` or `No`.