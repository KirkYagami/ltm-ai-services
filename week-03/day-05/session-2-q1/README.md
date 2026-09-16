# Loan Default Prediction Using L1 vs L2 Logistic Regression

## Project Overview

This project builds a machine learning pipeline for predicting loan defaults and comparing L1 and L2 regularization in Logistic Regression.

Two Logistic Regression models are trained:

- L1 (Lasso) regularized Logistic Regression
- L2 (Ridge) regularized Logistic Regression

The models are compared using:

- Classification Accuracy
- Number of non-zero model coefficients

This demonstrates how L1 regularization can perform feature selection by reducing some coefficients to zero.

---

## Project Structure

```text
Project/
├── loan_default.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_loan_data.csv
├── scaler.pkl
├── l1_model.pkl
└── l2_model.pkl
```

The last four files are generated when the program runs.

---

## Dataset

Input dataset:

```text
loan_default.csv
```

### Numerical Features

- `Age`
- `AnnualIncome`
- `LoanAmount`
- `CreditScore`
- `DebtToIncomeRatio`

### Categorical Features

- `EmploymentType`
- `LoanPurpose`

### Target

```text
Defaulted
```

Values:

- `1` = Defaulted
- `0` = Repaid

---

## Task 1: Data Preprocessing

Function:

```python
preprocess_data()
```

The function performs the following steps:

1. Loads `loan_default.csv`.
2. Automatically identifies numerical and categorical features.
3. Fills missing numerical values using the respective column mean.
4. Fills missing categorical values using the respective column mode.
5. Saves the cleaned dataset as:

```text
cleaned_loan_data.csv
```

6. Separates `Defaulted` from the feature columns.
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

## Task 2: Model Training

Function:

```python
train_models(X_train, y_train)
```

Two Logistic Regression models are trained.

### L1 Model

```python
LogisticRegression(
    penalty="l1",
    solver="saga",
    max_iter=5000,
    random_state=42
)
```

The model is saved as:

```text
l1_model.pkl
```

### L2 Model

```python
LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)
```

The model is saved as:

```text
l2_model.pkl
```

The function returns:

```python
l1_model, l2_model
```

---

## Task 3: Model Comparison

Function:

```python
compare_models(l1_model, l2_model, X_test, y_test)
```

Predictions are generated from both models.

Accuracy is calculated using:

```python
accuracy_score()
```

The number of non-zero coefficients is calculated from each model's:

```python
coef_
```

The function returns:

```python
l1_accuracy, l2_accuracy, l1_nonzero, l2_nonzero
```

---

## L1 vs L2 Regularization

### L1 Regularization

L1 adds an absolute-value penalty to the Logistic Regression objective.

It can force some model coefficients exactly to zero.

Therefore, L1 can perform implicit feature selection.

### L2 Regularization

L2 adds a squared-coefficient penalty.

It generally reduces coefficient magnitudes without forcing them exactly to zero.

Therefore, most or all features normally remain in the model.

---

## Generated Files

| File | Description |
|---|---|
| `cleaned_loan_data.csv` | Borrower data after missing-value imputation |
| `scaler.pkl` | Fitted StandardScaler object |
| `l1_model.pkl` | Trained L1 Logistic Regression model |
| `l2_model.pkl` | Trained L2 Logistic Regression model |

---

## Installation

Install dependencies using:

```bash
pip install -r installation.txt
```

Required packages include:

```text
pandas
numpy
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

The program will:

1. Preprocess the loan dataset.
2. Encode categorical features.
3. Standardize numerical features.
4. Split the dataset into training and test sets.
5. Train L1 and L2 Logistic Regression models.
6. Save both models.
7. Compare their accuracy and non-zero coefficients.

Example console output:

```text
L1 vs L2 Regularization Comparison:
L1 (Lasso) Accuracy: 0.9813
L2 (Ridge) Accuracy: 0.9817
L1 Non-zero Coefficients: 10 / 12
L2 Non-zero Coefficients: 12 / 12
```

Exact results depend on the supplied dataset.

---

## Running Tests

Run:

```bash
python3 -m pytest tests.py -v
```

For more verbose output:

```bash
python3 -m pytest tests.py -vv
```

The tests verify that:

- `preprocess_data()` is callable.
- Missing values are removed.
- `cleaned_loan_data.csv` is generated.
- `scaler.pkl` is generated.
- The train-test split is returned correctly.
- `train_models()` is callable.
- `l1_model.pkl` is generated.
- `l2_model.pkl` is generated.
- `compare_models()` is callable.
- Both accuracy values are between 0 and 1.
- L1 has no more non-zero coefficients than L2.