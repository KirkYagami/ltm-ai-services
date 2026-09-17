# Loan Approval Prediction — Outlier Impact on Gradient Boosting

## Objective

Build two Gradient Boosting classification models to investigate whether
IQR-based outlier capping changes loan approval prediction performance.

The experiment compares:

```text
Model 1 → Original / Raw Features
Model 2 → IQR-Capped Features
```

Both models use the same:

```text
Train/Test Split
Target
Gradient Boosting Configuration
Evaluation Metrics
```

This allows us to isolate the effect of outlier treatment.

---

# Dataset

```text
loan_applications.csv
```

Target:

```text
Approved
```

where:

```text
1 → Loan Approved
0 → Loan Not Approved
```

Features include:

```text
ApplicantIncome
LoanAmount
CreditScore
EmploymentYears
DebtToIncome
NumCreditLines
Age
PropertyValue
```

Missing values occur in:

```text
ApplicantIncome
CreditScore
EmploymentYears
```

---

# Complete Workflow

```text
loan_applications.csv
          │
          ▼
    Handle Missing Data
          │
          ▼
 Separate Features / Target
          │
          ▼
    80/20 Train-Test Split
    random_state = 42
    stratify = target
          │
          │
          ├──────────────────────────┐
          │                          │
          ▼                          ▼
      RAW PATH                 CAPPED PATH
          │                          │
          │                    Training Data
          │                          │
          │                          ▼
          │                    Calculate IQR
          │                          │
          │                          ▼
          │                   Learn IQR Bounds
          │                          │
          │                    ┌─────┴─────┐
          │                    ▼           ▼
          │               Cap Train    Cap Test
          │                    │       using SAME
          │                    │        bounds
          ▼                    ▼
   StandardScaler        StandardScaler
          │                    │
          ▼                    ▼
 Gradient Boosting      Gradient Boosting
          │                    │
          ▼                    ▼
      Evaluate              Evaluate
          │                    │
          └──────────┬─────────┘
                     ▼
             Compare Metrics
```

---

# Function 1 — `preprocess_data()`

```python
preprocess_data(filepath)
```

Returns:

```python
train_features
test_features
train_target
test_target
```

## Missing-Value Treatment

Numerical missing values are replaced using:

```python
median()
```

Conceptually:

```text
ApplicantIncome

45,000
52,000
NaN
60,000
500,000
```

The median is less affected by the extremely large observation than the mean.

---

# Train/Test Split

The project requires:

```python
train_test_split(
    features,
    target,
    test_size=0.20,
    random_state=42,
    stratify=target
)
```

Therefore:

```text
12,000 observations
       │
       ├── 80% → 9,600 Training
       │
       └── 20% → 2,400 Testing
```

`stratify=target` attempts to preserve the target-class proportions in both
partitions.

For example:

```text
Full Dataset
Approved=1 → 40%
Approved=0 → 60%

        ↓ stratified split

Training               Testing
1 → ~40%                1 → ~40%
0 → ~60%                0 → ~60%
```

---

# Function 2 — `cap_outliers()`

```python
cap_outliers(features)
```

This function performs IQR-based outlier detection and capping.

Returns:

```python
capped
outlier_counts
bounds
```

---

# Understanding IQR

IQR means:

```text
Interquartile Range
```

It describes the spread of the middle 50% of observations.

```text
Q1 = 25th percentile
Q3 = 75th percentile

IQR = Q3 - Q1
```

Example:

```text
Q1 = 40,000
Q3 = 80,000

IQR = 80,000 - 40,000

IQR = 40,000
```

---

# IQR Outlier Rule

The project uses:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Using:

```text
Q1  = 40,000
Q3  = 80,000
IQR = 40,000
```

we obtain:

```text
Lower Bound
= 40,000 - 1.5(40,000)
= -20,000

Upper Bound
= 80,000 + 1.5(40,000)
= 140,000
```

Therefore:

```text
ApplicantIncome

 30,000   → Normal
 70,000   → Normal
120,000   → Normal
250,000   → Outlier
```

because:

```text
250,000 > 140,000
```

---

# Outlier Detection

An observation is counted as an outlier when:

```python
value < lower_bound
```

OR:

```python
value > upper_bound
```

The function records these counts:

```python
outlier_counts[column] = count
```

Example:

```python
{
    "ApplicantIncome": 106,
    "LoanAmount": 110,
    "CreditScore": 60
}
```

---

# What Does Capping Mean?

The exercise does NOT delete outlier observations.

Instead, extreme values are clipped to the calculated limits.

Suppose:

```text
Lower Bound = 10,000
Upper Bound = 140,000
```

Then:

```text
Original      After Capping

  5,000   →      10,000
 30,000   →      30,000
 80,000   →      80,000
150,000   →     140,000
300,000   →     140,000
```

In Pandas:

```python
features[column].clip(
    lower=lower_bound,
    upper=upper_bound
)
```

So:

```text
Below lower bound → replace with lower bound

Inside bounds
→ unchanged

Above upper bound
→ replace with upper bound
```

---

# Why Cap Instead of Delete?

Deleting an outlier means:

```text
Observation disappears
```

Capping means:

```text
Observation remains
but extreme feature value is limited
```

For example, a loan applicant with an unusually high income still represents
a real applicant.

Instead of deleting the entire applicant:

```text
Applicant
Income = $2,000,000
Credit Score = 760
Age = 45
...
```

capping modifies only the extreme value.

Conceptually:

```text
$2,000,000

     ↓ cap

Upper IQR Bound
```

while preserving the rest of the observation.

Whether capping is appropriate in a real application depends on whether the
extreme value represents noise, an error, or a legitimate rare observation.

---

# Most Important Concept — Learn Bounds from Training Data

The correct sequence is:

```text
                TRAINING DATA
                     │
                     ▼
                  Find Q1
                     │
                  Find Q3
                     │
                  Find IQR
                     │
                     ▼
             Calculate Bounds
                     │
              ┌──────┴───────┐
              ▼              ▼
         Training Data    Test Data
              │              │
              ▼              ▼
        Apply Bounds     Apply SAME
                         Bounds
```

The test set must NOT calculate its own bounds.

---

# Why?

The test dataset represents unseen future data.

If we calculate:

```text
Q1
Q3
IQR
bounds
```

using the test dataset, information from the test distribution is being used
to influence preprocessing.

That creates:

```text
DATA LEAKAGE
```

The correct pattern is:

```python
capped_train, counts, bounds = cap_outliers(
    train_features
)

capped_test = apply_bounds(
    test_features,
    bounds
)
```

---

# Function 3 — `apply_bounds()`

```python
apply_bounds(features, bounds)
```

This function does NOT calculate IQR again.

It receives:

```python
bounds
```

learned from the training data.

Example:

```python
bounds = {
    "ApplicantIncome": (10000, 140000),
    "LoanAmount": (20000, 350000)
}
```

Then test observations are clipped using exactly these limits.

---

# Training vs Test Processing

Correct:

```text
TRAIN
  │
  ├── Calculate IQR
  ├── Calculate bounds
  └── Cap using those bounds
              │
              ▼
          Save Bounds
              │
              ▼
TEST ──► Apply saved training bounds
```

Incorrect:

```text
TRAIN ──► Calculate training bounds

TEST  ──► Calculate completely new test bounds
```

The second approach allows the held-out test distribution to influence the
preprocessing transformation.

---

# Function 4 — `compare_boosting()`

```python
compare_boosting(
    raw_train,
    raw_test,
    capped_train,
    capped_test,
    train_target,
    test_target
)
```

This function performs the actual experiment.

It trains:

```text
Gradient Boosting #1
→ Original data

Gradient Boosting #2
→ Capped data
```

---

# Experiment Design

```text
                 SAME TRAIN/TEST SPLIT
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
         RAW DATA                 CAPPED DATA
            │                         │
            ▼                         ▼
      StandardScaler             StandardScaler
        #1 fitted                 #2 fitted
            │                         │
            ▼                         ▼
 GradientBoostingClassifier  GradientBoostingClassifier
            │                         │
            ▼                         ▼
      Predictions                 Predictions
            │                         │
            ▼                         ▼
   Accuracy/Precision         Accuracy/Precision
      Recall/F1                 Recall/F1
            │                         │
            └────────────┬────────────┘
                         ▼
                    Comparison
```

---

# Why Two StandardScalers?

The exercise specifically requires a separate scaler for each experimental
pipeline.

Raw data has one distribution:

```text
Raw ApplicantIncome
```

while capped data has a modified distribution:

```text
Capped ApplicantIncome
```

Therefore their:

```text
mean
standard deviation
```

can differ.

We fit:

```python
raw_scaler.fit(raw_train)
```

for the raw model.

Separately:

```python
capped_scaler.fit(capped_train)
```

for the capped model.

---

# Correct Scaling Pattern

For raw data:

```python
raw_train_scaled = raw_scaler.fit_transform(
    raw_train
)

raw_test_scaled = raw_scaler.transform(
    raw_test
)
```

Notice:

```text
TRAIN → fit_transform()
TEST  → transform()
```

Never:

```text
TEST → fit_transform()
```

The same rule applies to capped data.

---

# Gradient Boosting Configuration

Both experiments use:

```python
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

Keeping these parameters identical is important.

Otherwise:

```text
Model performance difference
```

could come from:

```text
Outlier Treatment
        +
Different Hyperparameters
```

and we would no longer know which change caused the result.

---

# Controlled Experiment

The experiment should change essentially one preprocessing treatment:

```text
             MODEL A                MODEL B

Dataset       Same                   Same
Split         Same                   Same
Target        Same                   Same
Algorithm     Same                   Same
Estimators    100                    100
Depth         3                      3
Learning Rate 0.1                    0.1
Random State  42                     42

Outliers      Original               Capped
                ▲
                │
         Main comparison
```

---

# Evaluation Metrics

Both models are evaluated using:

```text
Accuracy
Precision
Recall
F1 Score
```

---

# Accuracy

```text
           Correct Predictions
Accuracy = ───────────────────
           Total Predictions
```

It measures overall classification correctness.

---

# Precision

```text
             TP
Precision = ───────
           TP + FP
```

Question answered:

```text
Of applications predicted as approved,
how many were actually approved?
```

---

# Recall

```text
          TP
Recall = ───────
         TP + FN
```

Question answered:

```text
Of applications that actually were approved,
how many did the model identify?
```

---

# F1 Score

```text
                 Precision × Recall
F1 = 2 × ─────────────────────────────
                 Precision + Recall
```

F1 combines precision and recall using their harmonic mean.

---

# Comparison Dictionary

The function returns a nested dictionary:

```python
{
    "Before Capping": {
        "Accuracy": ...,
        "Precision": ...,
        "Recall": ...,
        "F1 Score": ...
    },

    "After Capping": {
        "Accuracy": ...,
        "Precision": ...,
        "Recall": ...,
        "F1 Score": ...
    }
}
```

This makes the two experimental results easy to compare.

---

# Understanding the Expected Result

The example output shows approximately:

```text
Before Capping

Accuracy  = 0.6383
Precision = 0.6408
Recall    = 0.6376
F1        = 0.6392
```

and:

```text
After Capping

Accuracy  = 0.6379
Precision = 0.6401
Recall    = 0.6385
F1        = 0.6393
```

The difference is tiny.

That is itself useful.

Outlier treatment does NOT automatically guarantee a substantial improvement
in model performance.

The purpose of the experiment is to measure the impact rather than assume
that capping must help.

---

# Why Gradient Boosting May Be Less Sensitive to Outliers

Gradient Boosting here uses decision trees as its base learners.

Trees create rules such as:

```text
ApplicantIncome <= 74,500?
```

rather than relying directly on Euclidean distance.

Therefore, tree-based models are often less sensitive to feature magnitude
and extreme numerical values than algorithms such as:

```text
KNN
K-Means
Linear Regression
Logistic Regression in some optimization contexts
SVM
```

However, outliers can still influence:

```text
split locations
rare partitions
residual/error fitting
generalization
```

so their impact should be evaluated empirically.

---

# Important Note About StandardScaler

Gradient Boosted Trees generally do not require feature standardization.

A tree can split:

```text
ApplicantIncome < 75,000
```

and after scaling use an equivalent threshold:

```text
ApplicantIncome_scaled < 0.42
```

The ordering of observations remains unchanged.

However, this exercise explicitly requires:

```python
StandardScaler
```

so the implementation includes it.

---

# Saving the Model

The project saves the model trained on capped data:

```python
joblib.dump(
    capped_model,
    "approval_model.joblib"
)
```

Generated file:

```text
approval_model.joblib
```

---

# Run the Project

```bash
python3 main.py
```

Run tests:

```bash
python3 -m pytest tests.py -v
```

Expected tested components:

```text
preprocess_data()
    ✓ callable
    ✓ removes missing values
    ✓ train > test

cap_outliers()
    ✓ callable
    ✓ preserves shape
    ✓ returns counts
    ✓ returns bounds

apply_bounds()
    ✓ callable
    ✓ preserves shape

compare_boosting()
    ✓ callable
    ✓ returns comparison dictionary
    ✓ returns trained capped model

Output
    ✓ approval_model.joblib
```

---

# Core Concept

The most important idea in this exercise is not merely the IQR formula.

It is the separation between **learning preprocessing parameters** and
**applying preprocessing parameters**.

```text
                    TRAINING SET
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Learn IQR Bounds       Learn Scaler
              │                     │
              └──────────┬──────────┘
                         │
                  Learned Parameters
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
        Transform Train         Transform Test
```

General machine-learning rule:

```text
TRAINING DATA
→ FIT preprocessing

TEST / NEW DATA
→ TRANSFORM using fitted preprocessing
```

Examples:

```text
Median       → learn from train
IQR bounds   → learn from train
Scaler       → fit on train
Encoder      → learn categories from train
PCA          → fit on train
Feature selection → determine from train
```

The test set should simulate genuinely unseen data.

**Learn from training data. Apply to test data. Never let the test set teach the pipeline how to preprocess itself.**