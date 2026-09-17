# Credit Risk Assessment using SVM

This project demonstrates the effect of **regularization on a Support Vector Machine (SVM)** using an **RBF kernel**.

Two SVM classifiers are trained:

```text
C = 100
C = 0.01
```

The goal is to compare their training and testing performance and understand the relationship between:

```text
Regularization
      ↓
Model Complexity
      ↓
Bias / Variance
      ↓
Generalization
```

---

# 1. Problem Type

`Defaulted` is the target variable:

```text
Defaulted = 0 → Applicant did not default
Defaulted = 1 → Applicant defaulted
```

Therefore, this is a **binary classification problem**.

The model learns:

```text
Applicant Features
       ↓
       SVM
       ↓
Predicted Default Status
       ↓
      0 / 1
```

---

# 2. Data Preprocessing

Machine-learning algorithms require clean numerical input.

The original dataset contains both numerical and categorical features.

## Numerical Features

```text
AnnualIncome
CreditScore
LoanAmount
EmploymentYears
NumLatePayments
```

Their preprocessing pipeline is:

```text
Numerical Data
      ↓
Mean Imputation
      ↓
StandardScaler
      ↓
Scaled Numerical Features
```

## Categorical Features

```text
LoanPurpose
HomeOwnership
```

Their preprocessing pipeline is:

```text
Categorical Data
       ↓
Mode Imputation
       ↓
One-Hot Encoding
       ↓
Numerical Dummy Variables
```

`ColumnTransformer` allows these two pipelines to be applied to different columns simultaneously.

---

# 3. Missing Value Imputation

Machine-learning models generally cannot directly process missing feature values.

For numerical columns the project uses:

```python
SimpleImputer(strategy="mean")
```

Example:

```text
Income

50000
60000
NaN
70000
```

Mean:

```text
(50000 + 60000 + 70000) / 3
= 60000
```

The missing value becomes:

```text
50000
60000
60000
70000
```

For categorical columns:

```python
SimpleImputer(strategy="most_frequent")
```

The most frequently occurring category is used to replace missing values.

---

# 4. StandardScaler

Numerical features have very different scales.

For example:

```text
AnnualIncome      → 15,000 - 120,000
CreditScore       → 300 - 850
EmploymentYears   → 0 - 40
```

This matters significantly for SVM because distance between observations affects the RBF kernel.

The project therefore uses:

```python
StandardScaler()
```

which approximately transforms a feature using:

```math
z = (x - mean) / standard_deviation
```

After scaling, features are represented on comparable scales.

This prevents a large-valued feature such as income from dominating simply because of its numerical magnitude.

---

# 5. One-Hot Encoding

SVM cannot directly understand categories such as:

```text
LoanPurpose = Home
LoanPurpose = Auto
LoanPurpose = Medical
```

They must be converted into numerical features.

The project uses:

```python
OneHotEncoder(
    drop="first",
    handle_unknown="ignore"
)
```

For example:

```text
HomeOwnership

Own
Rent
Mortgage
```

could become:

```text
            Own    Rent
Mortgage     0      0
Own          1      0
Rent         0      1
```

One category is removed because:

```python
drop="first"
```

is enabled.

---

# 6. Train-Test Split

The processed dataset is divided into:

```text
Entire Dataset
      │
      ├──────────── 80% → Training Data
      │
      └──────────── 20% → Testing Data
```

Training data is used to learn the model.

Testing data represents unseen observations and is used to evaluate how well the learned model generalizes.

The project uses:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

`random_state=42` makes the random split reproducible.

---

# 7. Support Vector Machine

A Support Vector Machine attempts to construct a decision boundary that separates classes.

For a simple two-dimensional problem:

```text
Class 0       Decision Boundary       Class 1

 ○ ○ ○               |               ● ● ●
 ○ ○                  |                ● ●
  ○ ○                 |               ● ●
                      |
                 ← margin →
```

SVM attempts to find a boundary with a useful separation between the classes.

The observations most influential in determining this boundary are called **support vectors**.

---

# 8. RBF Kernel

The project uses:

```python
SVC(kernel="rbf")
```

RBF stands for:

**Radial Basis Function**

A linear SVM creates a linear boundary.

```text
○ ○ ○ | ● ● ●
○ ○ ○ | ● ● ●
```

But real datasets frequently have nonlinear relationships.

Conceptually, the RBF kernel allows SVM to produce nonlinear decision boundaries.

```text
○ ○ ○ ○
○       ○
○   ●   ○
○  ●●●  ○
○   ●   ○
○       ○
○ ○ ○ ○
```

A straight line would struggle with such a pattern, whereas an RBF SVM can represent a curved boundary.

---

# 9. The C Hyperparameter

`C` is one of the most important SVM hyperparameters.

It controls how strongly classification errors are penalized.

A useful intuition is:

```text
Large C
   ↓
Larger penalty for training errors
   ↓
Model tries harder to classify
training observations correctly
   ↓
Potentially more complex boundary
```

while:

```text
Small C
   ↓
Smaller penalty for training errors
   ↓
More violations/errors tolerated
   ↓
Stronger regularization
   ↓
Typically smoother/simpler boundary
```

Therefore:

| C | Regularization | Typical Effect |
|---|---|---|
| Large C | Weak | Lower bias, potentially higher variance |
| Small C | Strong | Higher bias, potentially lower variance |

These are tendencies, not guarantees for every dataset.

---

# 10. Why Compare C=100 and C=0.01?

The project deliberately compares two very different values.

## C = 100

```text
Large C
   ↓
Weak Regularization
   ↓
Training mistakes heavily penalized
   ↓
Model may fit training details closely
   ↓
Potential Higher Variance
```

This can produce excellent training performance while performing worse on unseen observations.

That is evidence consistent with **overfitting** when the train-test difference is substantial.

## C = 0.01

```text
Small C
   ↓
Strong Regularization
   ↓
Training mistakes more tolerated
   ↓
Simpler / smoother boundary
   ↓
Potential Lower Variance
```

Training accuracy may decrease, but test performance can sometimes improve.

---

# 11. Bias-Variance Trade-Off

Bias and variance describe two important sources of model error.

## High Bias

The model is too restrictive to capture important patterns.

```text
Model too simple
      ↓
Underfitting
      ↓
High Bias
```

## High Variance

The model responds too strongly to details or noise in the training dataset.

```text
Model too flexible
      ↓
Fits training details
      ↓
Poorer stability on unseen data
      ↓
High Variance
```

Regularization helps control model flexibility.

For this SVM experiment:

```text
C decreases
     ↓
Regularization increases
     ↓
Model becomes more constrained
     ↓
Variance tends to decrease
Bias tends to increase
```

Conversely:

```text
C increases
     ↓
Regularization decreases
     ↓
Model can fit training data more aggressively
     ↓
Bias tends to decrease
Variance tends to increase
```

---

# 12. Train-Test Accuracy Gap

The project calculates:

```text
Accuracy Gap =
|Training Accuracy - Testing Accuracy|
```

Suppose:

```text
Training Accuracy = 0.90
Testing Accuracy  = 0.82
```

Then:

```text
Gap = |0.90 - 0.82|
    = 0.08
```

A large gap can be evidence that the model fits its training observations considerably better than unseen observations.

A small gap indicates that train and test performance are similar, although a small gap **does not automatically mean the model is good**.

For example:

```text
Train Accuracy = 55%
Test Accuracy  = 54%
Gap            = 1%
```

The gap is tiny, but both performances are poor.

Therefore, always examine both:

```text
Absolute Test Performance
          +
Train-Test Performance Gap
```

---

# 13. Expected Experiment

The supplied example output demonstrates:

```text
C = 100

Train Accuracy ≈ 0.8971
Test Accuracy  ≈ 0.8290
Gap            ≈ 0.0681
```

Compared with:

```text
C = 0.01

Train Accuracy ≈ 0.8385
Test Accuracy  ≈ 0.8367
Gap            ≈ 0.0018
```

Conceptually:

```text
             C=100                 C=0.01
             ------                ------

Train        89.71%                83.85%
              │                      │
              │ 6.81%                │ 0.18%
              │                      │
Test         82.90%                83.67%
```

In this particular example, `C=100` fits the training data substantially more closely, while `C=0.01` produces much more similar train and test accuracies.

This demonstrates the central idea of the exercise:

> The model with the highest training accuracy is not necessarily the model that generalizes best.

---

# 14. Complete Workflow

```text
credit_risk_data.csv
        │
        ▼
     Pandas
        │
        ▼
Separate X and y
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
 Numerical               Categorical
 Features                  Features
        │                     │
 Mean Imputation        Mode Imputation
        │                     │
 StandardScaler         OneHotEncoder
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
          ColumnTransformer
                   │
                   ▼
          Processed Features
                   │
                   ▼
           Train/Test Split
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
    SVM C=100          SVM C=0.01
   Weak Reg.          Strong Reg.
          │                 │
          └────────┬────────┘
                   │
                   ▼
          Accuracy Comparison
                   │
                   ▼
          Train-Test Gap
```

---

# 15. Running the Project

Install dependencies:

```bash
pip install pandas scikit-learn joblib pytest
```

Run the main application:

```bash
python main.py
```

Run the automated tests:

```bash
pytest tests.py -v
```

The pytest suite imports and executes the required functions automatically, so `main.py` does not need to be run separately before running the tests.