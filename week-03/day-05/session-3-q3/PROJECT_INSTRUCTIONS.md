# Student Outcome Prediction — Early Stopping with SGDClassifier

## Objective

Build a binary classification model that predicts whether a student will pass or fail an enrolled course.

The project compares two `SGDClassifier` models:

- SGD without early stopping
- SGD with early stopping

The purpose is to understand how early stopping can reduce unnecessary training and potentially improve generalization.

---

# Dataset

Input:

```text
student_outcome_data.csv
```

Target:

```text
Passed
```

where:

```text
0 = Failed
1 = Passed
```

## Features

### Numerical

```text
StudyHoursPerWeek
AttendanceRate
AssignmentScore
QuizScore
PreviousGPA
```

### Categorical

```text
CourseLevel
LearningMode
```

---

# Task 1 — preprocess_data()

## Function Signature

```python
preprocess_data(filepath)
```

Load the dataset using Pandas.

Separate:

```text
X = Features
y = Passed
```

Build a `ColumnTransformer` for preprocessing.

### Numerical Pipeline

Apply:

```text
Numerical Features
        ↓
Mean Imputation
        ↓
StandardScaler
```

Use:

```python
SimpleImputer(strategy="mean")
StandardScaler()
```

### Categorical Pipeline

Apply:

```text
Categorical Features
        ↓
Mode Imputation
        ↓
One-Hot Encoding
```

Use:

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(drop="first")
```

Transform the features and save the fully processed dataset as:

```text
cleaned_student_data.csv
```

The saved CSV must:

- contain no missing values
- contain no object/string columns
- contain the `Passed` target column

Split the data:

```text
80% → Training
20% → Testing
```

using:

```python
random_state=42
```

Return:

```python
X_train, X_test, y_train, y_test, raw_data
```

---

# Task 2 — train_model()

## Function Signature

```python
train_model(X_train, y_train)
```

Train two `SGDClassifier` models.

## Model 1 — Without Early Stopping

Use:

```python
SGDClassifier(
    loss="log_loss",
    alpha=1e-10,
    max_iter=5000,
    tol=None,
    random_state=42
)
```

Setting:

```python
tol=None
```

prevents convergence-based stopping and therefore forces the estimator to run all `max_iter` epochs.

Save as:

```text
sgd_no_early_stop_model.joblib
```

## Model 2 — With Early Stopping

Use:

```python
SGDClassifier(
    loss="log_loss",
    alpha=1e-10,
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
```

The model reserves part of the training data as validation data.

Training can terminate when validation performance fails to improve sufficiently.

Save as:

```text
sgd_early_stop_model.joblib
```

Return:

```python
sgd_no_early_stop, sgd_early_stop
```

---

# Task 3 — evaluate_model()

## Function Signature

```python
evaluate_model(
    sgd_no_early_stop,
    sgd_early_stop,
    X_train,
    y_train,
    X_test,
    y_test
)
```

For both models calculate:

```text
Training Accuracy
Testing Accuracy
Iteration Count
Train-Test Accuracy Gap
```

Retrieve iteration count using:

```python
model.n_iter_
```

Return:

```python
{
    "no_early_stop": {
        "train": ...,
        "test": ...,
        "iterations": ...
    },
    "early_stop": {
        "train": ...,
        "test": ...,
        "iterations": ...
    }
}
```

---

# Generated Files

Running the project generates:

```text
cleaned_student_data.csv

sgd_no_early_stop_model.joblib
sgd_early_stop_model.joblib
```

---

# Project Structure

```text
Project/
│
├── student_outcome_data.csv
├── main.py
├── tests.py
├── PROJECT_INSTRUCTIONS.md
├── README.md
└── installation.txt
```

---

# Installation

```bash
pip install pandas scikit-learn joblib pytest
```

Run:

```bash
python main.py
```

Run tests:

```bash
pytest tests.py -v
```

`pytest` can be executed directly. You do not need to run `main.py` first because the fixtures import and call `preprocess_data()` and `train_model()` themselves.