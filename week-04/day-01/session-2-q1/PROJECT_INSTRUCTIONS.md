# Employee Attrition Prediction Using Bagging Ensemble Comparison

## Objective

Build a binary classification pipeline for predicting employee attrition.

The project compares:

```text
Decision Tree
     VS
Random Forest
```

using:

- Accuracy
- Precision
- Recall
- F1 Score

The model with the highest F1 Score is saved as:

```text
attrition_model.joblib
```

---

# Dataset

```text
employee_attrition.csv
```

Target:

```text
Attrition
```

where:

```text
0 → Employee stayed
1 → Employee left
```

Features include:

| Feature | Meaning |
|---|---|
| Age | Employee age |
| MonthlyIncome | Monthly income |
| YearsAtCompany | Company tenure |
| DistanceFromHome | Commute distance |
| JobSatisfaction | Satisfaction rating |
| OverTime | Yes / No |
| Department | Employee department |
| WorkLifeBalance | Work-life balance rating |

---

# Complete Pipeline

```text
employee_attrition.csv
          │
          ▼
      Load Data
          │
          ▼
 ┌────────────────────┐
 │ Handle Missing Data│
 └─────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
 Numerical   Categorical
  Median        Mode
     └─────┬─────┘
           ▼
 Separate Attrition
           │
           ▼
    One-Hot Encoding
     drop_first=True
           │
           ▼
   80% / 20% Split
      stratified
           │
           ▼
     StandardScaler
           │
           ▼
      ┌────┴────┐
      ▼         ▼
 Decision     Random
   Tree       Forest
      │         │
      └────┬────┘
           ▼
      Predictions
           │
           ▼
 Accuracy / Precision
 Recall / F1 Score
           │
           ▼
    Highest F1 Score
           │
           ▼
 attrition_model.joblib
```

---

# 1. Missing-Value Handling

The dataset contains missing values in numerical columns such as:

```text
MonthlyIncome
YearsAtCompany
DistanceFromHome
```

Numerical missing values are filled using the:

```text
MEDIAN
```

Example:

```text
3000
4000
5000
NaN
6000
```

Median:

```text
5000
```

Therefore:

```text
NaN → 5000
```

Code:

```python
data[column] = data[column].fillna(
    data[column].median()
)
```

---

# Categorical Missing Values

Categorical columns include:

```text
OverTime
Department
```

Missing categorical values are filled using the:

```text
MODE
```

The mode is the most frequently occurring category.

Example:

```text
Engineering
Sales
Engineering
NaN
Engineering
```

Mode:

```text
Engineering
```

Therefore:

```text
NaN → Engineering
```

---

# 2. Separate Features and Target

Target:

```python
target = data["Attrition"]
```

Features:

```python
features = data.drop(
    columns=["Attrition"]
)
```

Conceptually:

```text
Employee Data
      │
      ├──────────────► X = Features
      │
      └──────────────► y = Attrition
```

`Attrition` must not be included in the input features.

---

# 3. One-Hot Encoding

Machine-learning models require numerical inputs.

Categorical values such as:

```text
Department

Engineering
HR
Marketing
Sales
Support
```

must therefore be encoded.

The project uses:

```python
pd.get_dummies(
    features,
    drop_first=True
)
```

Possible result:

```text
Department_HR
Department_Marketing
Department_Sales
Department_Support
```

If all of these are:

```text
0
```

the observation belongs to the dropped reference category.

---

# Why `drop_first=True`?

Suppose:

```text
OverTime

Yes
No
```

Without dropping:

```text
OverTime_No
OverTime_Yes
```

These columns contain redundant information because:

```text
OverTime_No = 1 - OverTime_Yes
```

With:

```python
drop_first=True
```

only one dummy variable is required.

---

# 4. Train/Test Split

The project requires:

```text
80% Training
20% Testing
```

using:

```python
train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)
```

For 12,000 observations:

```text
12,000
   │
   ├── 9,600 → Training
   │
   └── 2,400 → Testing
```

---

# Why `random_state=42`?

Random splitting normally produces a different split on different runs.

Setting:

```python
random_state=42
```

makes the split reproducible.

```text
Run 1 ─┐
Run 2 ─┼──► Same split
Run 3 ─┘
```

The number `42` itself has no special machine-learning meaning.

---

# Why `stratify=target`?

Suppose the complete dataset contains:

```text
Attrition = 0 → 70%
Attrition = 1 → 30%
```

A stratified split attempts to maintain approximately the same distribution:

```text
TRAIN
0 → 70%
1 → 30%

TEST
0 → 70%
1 → 30%
```

This is especially useful for classification problems.

---

# 5. Feature Scaling

The project explicitly requires:

```python
StandardScaler()
```

Standardization calculates:

```text
             x - μ
z = ─────────────────────
               σ
```

where:

```text
x = original value
μ = training feature mean
σ = training feature standard deviation
```

After standardization, features are approximately centered around:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

---

# VERY IMPORTANT — Prevent Data Leakage

The scaler learns parameters from the **training set only**.

Correct:

```python
scaler.fit_transform(train_features)
scaler.transform(test_features)
```

Flow:

```text
TRAINING DATA
     │
     ▼
   FIT
     │
     ▼
Learn μ and σ
     │
     ├──────────► transform TRAIN
     │
     └──────────► transform TEST
```

Incorrect:

```python
scaler.fit_transform(test_features)
```

because this allows information from the test dataset to influence preprocessing.

The test set should simulate unseen data.

---

# Why Convert Back to DataFrame?

`StandardScaler` returns a NumPy array.

The specification requires:

```text
train_features → pandas.DataFrame
test_features  → pandas.DataFrame
```

Therefore:

```python
train_features = pd.DataFrame(
    train_scaled,
    columns=train_features.columns,
    index=train_features.index
)
```

This preserves meaningful feature names.

---

# Function 1 — `preprocess_data()`

Signature:

```python
preprocess_data(filepath)
```

Returns exactly:

```python
(
    train_features,
    test_features,
    train_target,
    test_target
)
```

Pipeline:

```text
CSV
 ↓
Imputation
 ↓
X / y
 ↓
One-Hot Encoding
 ↓
80/20 Split
 ↓
Scaling
 ↓
Return Four Objects
```

---

# Function 2 — `train_models()`

Signature:

```python
train_models(
    train_features,
    train_target
)
```

Two classifiers are trained.

## Decision Tree

```python
DecisionTreeClassifier(
    random_state=42
)
```

A decision tree learns hierarchical decision rules.

Simplified example:

```text
             OverTime?
            /         \
          Yes          No
          /             \
 JobSatisfaction?     Stay
     /       \
   Low       High
    │          │
  Leave      Stay
```

---

# Random Forest

The second model is:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

A Random Forest creates many decision trees.

```text
              DATA
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
     Tree 1  Tree 2   Tree 3 ... Tree 100
       │       │       │
       └───────┼───────┘
               ▼
         Majority Vote
               ▼
       Final Prediction
```

---

# Why Is Random Forest a Bagging Ensemble?

Random Forest uses the principle of:

```text
BAGGING
```

meaning:

```text
Bootstrap
   +
Aggregating
```

Each tree is trained using a bootstrap sample of the training observations.

Conceptually:

```text
Original Training Dataset
          │
   ┌──────┼──────┐
   ▼      ▼      ▼
Sample 1 Sample 2 Sample 3
   │      │      │
 Tree 1 Tree 2 Tree 3
   │      │      │
   └──────┼──────┘
          ▼
      Aggregate
          ▼
   Final Prediction
```

Random Forest additionally introduces randomness when selecting candidate features at tree splits.

---

# Why Multiple Trees?

A single Decision Tree can have high variance.

Small changes in training observations can sometimes produce substantially different trees.

Random Forest combines many trees:

```text
Many diverse trees
        ↓
Aggregate predictions
        ↓
More stable prediction
```

This is one of the central ideas behind bagging.

---

# `train_models()` Return Value

The trained models are stored in:

```python
models = {
    "Decision Tree": decision_tree,
    "Random Forest": random_forest
}
```

Therefore:

```text
models
 │
 ├── "Decision Tree"
 │        ↓
 │    trained DT
 │
 └── "Random Forest"
          ↓
      trained RF
```

---

# Function 3 — `evaluate_models()`

Signature:

```python
evaluate_models(
    models,
    test_features,
    test_target
)
```

For each model:

```text
Test Features
     ↓
model.predict()
     ↓
Predictions
     ↓
Compare with y_test
     ↓
Metrics
```

Four metrics are calculated:

```text
Accuracy
Precision
Recall
F1 Score
```

---

# Confusion Matrix Foundation

For binary classification:

```text
                    ACTUAL
                 1          0

PREDICTED  1     TP         FP

           0     FN         TN
```

For this problem:

```text
TP
Employee actually leaves
and model predicts leave

FP
Employee stays
but model predicts leave

FN
Employee leaves
but model predicts stay

TN
Employee stays
and model predicts stay
```

---

# Accuracy

```text
             TP + TN
Accuracy = ─────────────
           TP+TN+FP+FN
```

It measures:

> What fraction of all predictions were correct?

---

# Precision

```text
              TP
Precision = ───────
            TP + FP
```

It answers:

> Of employees predicted to leave, how many actually left?

High precision means fewer false alarms.

---

# Recall

```text
           TP
Recall = ───────
         TP + FN
```

It answers:

> Of all employees who actually left, how many did the model detect?

High recall means fewer attrition cases are missed.

---

# F1 Score

F1 combines Precision and Recall:

```text
                  Precision × Recall
F1 = 2 × ─────────────────────────────
                  Precision + Recall
```

Example:

```text
Precision = 0.80
Recall    = 0.60
```

Then:

```text
          0.80 × 0.60
F1 = 2 × ─────────────
          0.80 + 0.60

   ≈ 0.686
```

F1 becomes high when Precision and Recall are both reasonably high.

---

# Evaluation Result Structure

`evaluate_models()` returns a nested dictionary:

```python
{
    "Decision Tree": {
        "Accuracy": ...,
        "Precision": ...,
        "Recall": ...,
        "F1 Score": ...
    },

    "Random Forest": {
        "Accuracy": ...,
        "Precision": ...,
        "Recall": ...,
        "F1 Score": ...
    }
}
```

This structure allows access such as:

```python
results["Random Forest"]["F1 Score"]
```

---

# Selecting the Model by F1

The project requires selecting the model having the highest:

```text
F1 Score
```

The code uses:

```python
best_model_name = max(
    results,
    key=lambda name:
        results[name]["F1 Score"]
)
```

Conceptually:

```text
Decision Tree F1 ──┐
                   ├── Compare ──► Highest F1
Random Forest F1 ──┘
```

---

# Saving the Model

The selected estimator is persisted using:

```python
joblib.dump(
    models[best_model_name],
    "attrition_model.joblib"
)
```

Result:

```text
attrition_model.joblib
```

`joblib` serializes the trained Python model so that it can later be loaded instead of retraining it.

Example:

```python
model = joblib.load(
    "attrition_model.joblib"
)
```

---

# Important Practical Note About This Exercise

Decision Trees and Random Forests generally **do not require feature standardization** because their splits depend on feature thresholds rather than Euclidean distance.

However, this project explicitly requires `StandardScaler`, so scaling must be performed to satisfy the specified preprocessing pipeline.

This differs from algorithms such as:

```text
KNN
SVM
Logistic Regression
PCA
```

where feature scaling can be much more important.

---

# Expected Project Structure

```text
Project/
│
├── employee_attrition.csv
├── main.py
├── tests.py
├── installation.txt
└── attrition_model.joblib
```

The `.joblib` file appears after running the main program or after the corresponding test saves the selected model.

---

# Run Project

Install:

```bash
pip install pandas scikit-learn joblib pytest
```

Run:

```bash
python3 main.py
```

Run tests:

```bash
python3 -m pytest tests.py -v
```

---

# Core Concepts to Remember

```text
Median
  → Numerical missing values

Mode
  → Categorical missing values

One-Hot Encoding
  → Categories → numerical features

Stratification
  → Preserve class distribution

StandardScaler
  → Standardize feature scales

Decision Tree
  → Single hierarchical classifier

Random Forest
  → Ensemble of many randomized trees

Bagging
  → Bootstrap + aggregation

Precision
  → Correctness of positive predictions

Recall
  → Ability to find actual positives

F1
  → Harmonic mean of precision and recall

joblib
  → Persist trained model
```

The overall learning objective is:

> **Compare a single Decision Tree with a bagging-based Random Forest on the same train/test data, evaluate both using classification metrics, and persist the model selected according to F1 Score.**