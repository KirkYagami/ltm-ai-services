# Titanic Survival Prediction Using Decision Tree Classification

## Objective

Build a Decision Tree classification system that predicts whether a
Titanic passenger survived.

The project compares two Decision Tree models:

- Shallow tree with `max_depth=3`
- Deep tree with `max_depth=10`

The models use the Gini impurity criterion.

## Project Structure

```text
Project/
├── titanic.csv
├── new_passengers.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_titanic_data.csv
├── decision_tree_depth_3.joblib
├── decision_tree_depth_10.joblib
└── predicted_passengers.csv
```

## Training Dataset

The training file is:

```text
titanic.csv
```

Important columns include:

- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

The target column is:

```text
Survived
```

where:

```text
0 = Not Survived
1 = Survived
```

## Prediction Dataset

The prediction file is:

```text
new_passengers.csv
```

It contains passenger information but does not contain the `Survived`
target column.

---

## Preprocessing

Implement:

```python
preprocess_data(filepath)
```

Load the CSV file into a pandas DataFrame.

Drop the following irrelevant columns:

```text
PassengerId
Name
Ticket
Cabin
```

The `Survived` column must be separated from the feature set.

Handle missing values as follows:

```text
Age       -> median
Embarked  -> mode
```

Encode the categorical columns:

```text
Sex
Embarked
```

using one-hot encoding with:

```python
pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)
```

The processed feature DataFrame must contain no missing values.

For the training dataset, save the processed data as:

```text
cleaned_titanic_data.csv
```

Return:

```text
features
target
```

where both are pandas objects.

---

## Model Training

Implement:

```python
train_model(features, target, max_depth)
```

Split the dataset into 80% training data and 20% testing data.

Use:

```python
train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)
```

Create a Decision Tree classifier using:

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=max_depth,
    random_state=42
)
```

Train the model on the training set.

Generate predictions on the test set.

Calculate accuracy using:

```python
accuracy_score()
```

Round the accuracy to four decimal places.

Generate a classification report containing:

```text
precision
recall
f1-score
support
```

Return:

```text
classifier
accuracy
report
```

Save the model using joblib.

For `max_depth=3`, save:

```text
decision_tree_depth_3.joblib
```

For `max_depth=10`, save:

```text
decision_tree_depth_10.joblib
```

---

## Prediction

Implement:

```python
predict_passengers(
    classifier,
    feature_columns,
    filepath
)
```

Load:

```text
new_passengers.csv
```

Preserve the original input DataFrame.

Drop:

```text
PassengerId
Name
Ticket
Cabin
```

Handle missing values using the same strategy as training:

```text
Age       -> median
Embarked  -> mode
```

Apply one-hot encoding to:

```text
Sex
Embarked
```

using:

```python
drop_first=True
```

The encoded prediction features must be aligned with the training
feature columns.

Use:

```python
reindex(
    columns=feature_columns,
    fill_value=0
)
```

Generate predictions using the trained classifier.

Map predictions as:

```text
1 -> Survived
0 -> Not Survived
```

Add the prediction column:

```text
Predicted_Survival
```

The original columns must be preserved.

Save the result as:

```text
predicted_passengers.csv
```

Return the resulting DataFrame.

---

## Main Execution

The main block must:

1. Call `preprocess_data("titanic.csv")`.
2. Train a shallow tree with `max_depth=3`.
3. Print its accuracy.
4. Print its classification report.
5. Train a deep tree with `max_depth=10`.
6. Print its accuracy.
7. Print its classification report.
8. Use the deep tree for predictions.
9. Call `predict_passengers()`.
10. Save predictions to `predicted_passengers.csv`.
11. Print the prediction results.

---

## Expected Generated Files

After running the program, these files should exist:

```text
cleaned_titanic_data.csv
decision_tree_depth_3.joblib
decision_tree_depth_10.joblib
predicted_passengers.csv
```

The prediction file must contain:

```text
Predicted_Survival
```

and prediction values must be:

```text
Survived
Not Survived
```

There must be no null prediction values.

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

The program should be executed before running tests because some tests
verify that generated files exist.

---

## Expected Console Output

The output should contain information similar to:

```text
Shallow Tree (max_depth=3) Accuracy: 0.7681
Classification Report:

Deep Tree (max_depth=10) Accuracy: 0.7510
Classification Report:

Predictions saved to predicted_passengers.csv
```

The exact accuracy values depend on the supplied dataset.

## Important Requirements

- Use `DecisionTreeClassifier`.
- Use `criterion="gini"`.
- Use `random_state=42`.
- Use an 80/20 train-test split.
- Train models with depths 3 and 10.
- Handle missing Age using the median.
- Handle missing Embarked using the mode.
- Drop PassengerId, Name, Ticket, and Cabin.
- Use one-hot encoding with `drop_first=True`.
- Save both trained models using joblib.
- Save the cleaned training data.
- Save prediction results to `predicted_passengers.csv`.
- Preserve all original prediction-data columns.
- Add exactly one `Predicted_Survival` column.
- Ensure predictions contain no null values.