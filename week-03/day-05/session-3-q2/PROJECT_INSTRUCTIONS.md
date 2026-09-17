# Credit Risk Assessment — SVM Regularization

## Objective

Build a binary classification system that predicts whether a loan applicant will default.

The project compares two Support Vector Machine models using an RBF kernel:

- `C = 100` → weak regularization
- `C = 0.01` → strong regularization

The purpose is to observe how the SVM `C` hyperparameter affects model complexity, overfitting, and generalization.

---

## Dataset

Input file:

```text
credit_risk_data.csv
```

### Columns

| Column | Type | Description |
|---|---|---|
| AnnualIncome | Numerical | Applicant annual income |
| CreditScore | Numerical | Credit score |
| LoanAmount | Numerical | Requested loan amount |
| EmploymentYears | Numerical | Years of employment |
| NumLatePayments | Numerical | Previous late payments |
| LoanPurpose | Categorical | Auto, Education, Home, Medical, Personal |
| HomeOwnership | Categorical | Own, Mortgage, Rent |
| Defaulted | Binary | 0 = No Default, 1 = Default |

---

# Task 1 — `preprocess_data`

## Function

```python
preprocess_data(filepath)
```

The function must:

1. Load the CSV dataset using Pandas.
2. Separate `Defaulted` from the input features.
3. Identify the numerical columns:

```text
AnnualIncome
CreditScore
LoanAmount
EmploymentYears
NumLatePayments
```

4. Identify the categorical columns:

```text
LoanPurpose
HomeOwnership
```

5. Build a preprocessing pipeline using `ColumnTransformer`.

For numerical columns:

```text
Missing values
      ↓
Mean Imputation
      ↓
StandardScaler
```

For categorical columns:

```text
Missing values
      ↓
Most-Frequent / Mode Imputation
      ↓
OneHotEncoder(drop="first")
```

6. Transform the feature matrix.
7. Save the fully processed dataset as:

```text
cleaned_credit_data.csv
```

The saved dataset must:

- contain no missing values
- contain no object/string columns
- contain the `Defaulted` target column

8. Split the transformed data into:

```text
80% training data
20% testing data
```

Use:

```python
random_state=42
```

9. Return:

```python
X_train, X_test, y_train, y_test, raw_data
```

---

# Task 2 — `train_model`

## Function

```python
train_model(X_train, y_train)
```

Create two SVM classifiers.

### Model 1

```python
SVC(
    kernel="rbf",
    C=100,
    random_state=42
)
```

This represents relatively weak regularization.

### Model 2

```python
SVC(
    kernel="rbf",
    C=0.01,
    random_state=42
)
```

This represents strong regularization.

Fit both models using the training data.

Save them as:

```text
svm_c100_model.joblib
svm_c001_model.joblib
```

Return:

```python
svm_weak_reg, svm_strong_reg
```

---

# Task 3 — `evaluate_model`

## Function

```python
evaluate_model(
    svm_weak_reg,
    svm_strong_reg,
    X_train,
    y_train,
    X_test,
    y_test
)
```

For each model:

1. Predict the training observations.
2. Predict the testing observations.
3. Calculate training accuracy.
4. Calculate testing accuracy.
5. Compare the train-test accuracy gap.

Return:

```python
{
    "weak_reg": {
        "train": ...,
        "test": ...
    },
    "strong_reg": {
        "train": ...,
        "test": ...
    }
}
```

---

# Expected Project Structure

```text
Project/
│
├── credit_risk_data.csv
├── main.py
├── tests.py
├── PROJECT_INSTRUCTIONS.md
├── README.md
└── installation.txt
```

After running the program:

```text
Project/
│
├── cleaned_credit_data.csv
├── svm_c100_model.joblib
└── svm_c001_model.joblib
```

will also be generated.

---

# Installation

Install the dependencies:

```bash
pip install pandas scikit-learn joblib pytest
```

Run the program:

```bash
python main.py
```

Run the tests:

```bash
pytest tests.py -v
```

---

# Important

The test suite imports the functions directly from `main.py`.

Therefore, running `main.py` manually before executing `pytest` is not required.

For example:

```python
from main import preprocess_data
```

imports the function without executing the code inside:

```python
if __name__ == "__main__":
```

The pytest fixtures then call the required functions themselves.