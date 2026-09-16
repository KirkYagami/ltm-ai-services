# Student Performance Prediction Using GridSearchCV with KNN

## Project Overview

This project builds a machine learning pipeline for predicting student performance using the K-Nearest Neighbors (KNN) classification algorithm.

`GridSearchCV` is used to tune the number of neighbors and select the best KNN model using 5-fold cross-validation.

The pipeline performs:

- Missing value handling
- Categorical feature encoding
- Numerical feature standardization
- Train-test splitting
- KNN hyperparameter tuning
- 5-fold cross-validation
- Test-set evaluation

---

## Project Structure

```text
Project/
├── student_performance.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_student_data.csv
├── scaler.pkl
└── best_knn_model.pkl
```

The last three files are generated when the program runs.

---

## Dataset

Input dataset:

```text
student_performance.csv
```

### Numerical Features

- `Age`
- `StudyHoursPerWeek`
- `AttendanceRate`
- `PreviousGPA`

### Categorical Features

- `ParentEducation`
- `StudyMethod`

### Target

```text
Passed
```

Values:

- `1` = Passed
- `0` = Failed

---

## Task 1: Data Preprocessing

Function:

```python
preprocess_data()
```

The function:

1. Loads `student_performance.csv`.
2. Automatically identifies numerical and categorical columns.
3. Fills missing numerical values using the respective column mean.
4. Fills missing categorical values using the respective column mode.
5. Saves the cleaned dataset as:

```text
cleaned_student_data.csv
```

6. Separates `Passed` from the input features.
7. One-hot encodes categorical features.
8. Standardizes numerical features using `StandardScaler`.
9. Saves the fitted scaler as:

```text
scaler.pkl
```

10. Performs an 80/20 train-test split using:

```python
random_state=42
stratify=y
```

The function returns:

```python
X_train, X_test, y_train, y_test
```

---

## Task 2: KNN Hyperparameter Tuning

Function:

```python
tune_model(X_train, y_train)
```

A `KNeighborsClassifier` is created and tuned using `GridSearchCV`.

The parameter grid is:

```python
{
    "n_neighbors": [3, 5, 7, 9, 11]
}
```

Grid search uses:

```python
cv=5
scoring="accuracy"
```

This means every candidate value of `n_neighbors` is evaluated using 5-fold cross-validation on the training data.

The best-performing model is obtained from:

```python
grid_search.best_estimator_
```

The best parameters are obtained from:

```python
grid_search.best_params_
```

The best cross-validation score is obtained from:

```python
grid_search.best_score_
```

The selected model is saved as:

```text
best_knn_model.pkl
```

The function returns:

```python
best_model, best_params, best_cv_score
```

---

## Task 3: Results Reporting

Function:

```python
report_results(
    best_model,
    best_params,
    best_cv_score,
    X_test,
    y_test
)
```

The selected KNN model generates predictions on the held-out test set.

Test accuracy is calculated using:

```python
accuracy_score()
```

The function prints:

- Best hyperparameters
- Best cross-validation score
- Test accuracy

It returns:

```python
test_accuracy
```

---

## Generated Files

| File | Description |
|---|---|
| `cleaned_student_data.csv` | Student data after missing-value imputation |
| `scaler.pkl` | Fitted StandardScaler object |
| `best_knn_model.pkl` | Best KNN model selected by GridSearchCV |

---

## Understanding GridSearchCV

The project tests five possible values for K:

```text
3, 5, 7, 9, 11
```

For each value, 5-fold cross-validation is performed.

Conceptually:

```text
K = 3  ──> 5-fold CV ──> Mean Accuracy
K = 5  ──> 5-fold CV ──> Mean Accuracy
K = 7  ──> 5-fold CV ──> Mean Accuracy
K = 9  ──> 5-fold CV ──> Mean Accuracy
K = 11 ──> 5-fold CV ──> Mean Accuracy
                         │
                         ▼
                  Highest CV Score
                         │
                         ▼
                    Best KNN Model
```

The test set is not involved in selecting the best value of K. It is used afterward to evaluate the selected model.

---

## Installation

Install dependencies using:

```bash
pip install -r installation.txt
```

Required packages include:

```text
pandas
scikit-learn
joblib
pytest
```

---

## Running the Project

Navigate to the project directory:

```bash
cd Project
```

Run:

```bash
python3 main.py
```

The program will preprocess the dataset, perform GridSearchCV, save the best KNN model, and evaluate it on the test set.

Example console output:

```text
GridSearchCV Results:
Best Parameters: {'n_neighbors': 5}
Best CV Score: 0.9817
Test Accuracy: 0.9803
```

Exact results depend on the supplied dataset.

---

## Running Tests

Run:

```bash
python3 -m pytest tests.py -v
```

For additional verbosity:

```bash
python3 -m pytest tests.py -vv
```

The tests verify that:

- `preprocess_data()` is callable.
- Missing values are removed.
- `cleaned_student_data.csv` is generated.
- `scaler.pkl` is generated.
- Train-test data is returned correctly.
- `tune_model()` is callable.
- `best_knn_model.pkl` is generated.
- The returned parameters contain `n_neighbors`.
- The cross-validation score is between 0 and 1.
- `report_results()` is callable.
- Test accuracy is between 0 and 1.