# Healthcare Patient Risk Prediction
## KNN Underfitting vs Overfitting

## Project Overview

This project demonstrates **underfitting, overfitting, and the bias-variance trade-off** using the K-Nearest Neighbors (KNN) classification algorithm.

Two deliberately extreme values of `k` are used:

- `k = 1` → highly complex model, demonstrating overfitting/high variance
- `k = 50` → smoother model, demonstrating underfitting/high bias

The models predict whether a patient belongs to a high-risk group based on demographic, clinical, and lifestyle information.

---

## Project Structure

```text
Project/
├── patient_risk_data.csv
├── main.py
├── tests.py
└── installation.txt
```

Running the program also generates:

```text
cleaned_patient_data.csv
knn_k1_model.joblib
knn_k50_model.joblib
```

---

# Dataset

Input file:

```text
patient_risk_data.csv
```

The dataset contains approximately 15,000 patient records.

## Numerical Features

- `Age`
- `BMI`
- `BloodPressure`
- `Cholesterol`
- `HeartRate`

## Categorical Features

- `SmokingStatus`
- `ExerciseFrequency`

## Target

```text
AtRisk
```

Where:

```text
0 = Low Risk
1 = High Risk
```

The feature columns contain missing values that must be handled before model training.

---

# Task 1: Data Preprocessing

Function:

```python
preprocess_data(filepath)
```

The function loads the patient dataset and separates the features from the `AtRisk` target.

A preprocessing pipeline is created using `ColumnTransformer`.

## Numerical Pipeline

Numerical features undergo:

```text
Missing Values
      ↓
Mean Imputation
      ↓
StandardScaler
```

Mean imputation is performed using:

```python
SimpleImputer(strategy="mean")
```

Feature scaling is performed using:

```python
StandardScaler()
```

Scaling is particularly important for KNN because KNN calculates distances between observations.

Without scaling, features having large numerical ranges could dominate the distance calculation.

---

## Categorical Pipeline

Categorical features undergo:

```text
Missing Values
      ↓
Mode Imputation
      ↓
One-Hot Encoding
```

Mode imputation is performed using:

```python
SimpleImputer(strategy="most_frequent")
```

Categorical values are converted to numerical features using:

```python
OneHotEncoder(handle_unknown="ignore")
```

---

## Cleaned Dataset

After preprocessing, the transformed features are saved as:

```text
cleaned_patient_data.csv
```

The resulting dataset:

- contains no missing values
- contains no object/string feature columns
- contains encoded categorical features
- contains scaled numerical features
- retains the `AtRisk` target

---

## Train-Test Split

The dataset is divided using:

```python
train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42
)
```

Therefore:

```text
80% → Training Data
20% → Testing Data
```

The function returns:

```python
X_train,
X_test,
y_train,
y_test,
raw_data
```

---

# Task 2: Train KNN Models

Function:

```python
train_model(X_train, y_train)
```

Two KNN classifiers are created.

## Model 1 — k = 1

```python
KNeighborsClassifier(
    n_neighbors=1
)
```

With `k=1`, each prediction depends entirely on the nearest training observation.

This produces a highly flexible decision boundary.

The model can therefore effectively memorize the training data.

Typical behavior:

```text
Training Accuracy → Extremely High
Test Accuracy     → Significantly Lower
```

This demonstrates:

```text
OVERFITTING
High Variance
Low Bias
```

The model is saved as:

```text
knn_k1_model.joblib
```

---

# Model 2 — k = 50

```python
KNeighborsClassifier(
    n_neighbors=50
)
```

Here, each prediction considers 50 neighboring observations.

The decision boundary becomes much smoother because individual observations have considerably less influence.

An excessively large `k` can therefore make the model too simple to capture important patterns.

Typical behavior:

```text
Training Accuracy → Lower
Test Accuracy     → Similar and also relatively low
```

This demonstrates:

```text
UNDERFITTING
High Bias
Low Variance
```

The model is saved as:

```text
knn_k50_model.joblib
```

---

# Task 3: Model Evaluation

Function:

```python
evaluate_model(
    knn_k1,
    knn_k50,
    X_train,
    y_train,
    X_test,
    y_test
)
```

The function calculates both training and testing accuracy for each model.

Accuracy is calculated using:

```python
accuracy_score()
```

The returned dictionary has the structure:

```python
{
    "k1": {
        "train": k1_train_accuracy,
        "test": k1_test_accuracy
    },

    "k50": {
        "train": k50_train_accuracy,
        "test": k50_test_accuracy
    }
}
```

---

# Understanding the Accuracy Gap

The project also examines:

```text
Accuracy Gap = Training Accuracy - Test Accuracy
```

For example:

```text
k = 1

Train Accuracy = 1.0000
Test Accuracy  = 0.6420

Gap = 1.0000 - 0.6420
    = 0.3580
```

A large train-test gap is a classic indication that the model is fitting the training observations much better than unseen observations.

---

# Expected Behavior

The assignment demonstrates two extremes.

```text
             LOW k                         HIGH k
               │                              │
               ▼                              ▼
        Complex Model                  Simpler Model
               │                              │
               ▼                              ▼
        Low Training Error             Higher Training Error
               │                              │
               ▼                              ▼
         High Variance                    High Bias
               │                              │
               ▼                              ▼
         OVERFITTING                    UNDERFITTING

              k=1                           k=50
```

In practice, we would normally search for an intermediate value of `k` that generalizes better.

For example:

```text
k = 1       → Too complex
k = 50      → Too simple

Some intermediate k
      ↓
Better bias-variance balance
```

Techniques such as cross-validation and `GridSearchCV` can be used to select that value.

---

# Generated Files

| File | Description |
|---|---|
| `cleaned_patient_data.csv` | Fully preprocessed patient dataset |
| `knn_k1_model.joblib` | Trained KNN model using k=1 |
| `knn_k50_model.joblib` | Trained KNN model using k=50 |

---

# Installation

Install the required dependencies:

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

# Running the Program

Navigate to the project directory:

```bash
cd Project
```

Run:

```bash
python3 main.py
```

The program will:

1. Load the patient dataset.
2. Display the original missing-value counts.
3. Impute missing values.
4. Encode categorical features.
5. Standardize numerical features.
6. Save the cleaned dataset.
7. Create an 80/20 train-test split.
8. Train KNN with `k=1`.
9. Train KNN with `k=50`.
10. Save both models.
11. Calculate train and test accuracy.
12. Display the train-test accuracy gap.

Example:

```text
KNN Model Comparison: Overfitting vs Underfitting
================================================

k=1 (Overfitting - High Variance):
  Train Accuracy: 1.0000
  Test Accuracy:  0.6420
  Accuracy Gap:   0.3580

k=50 (Underfitting - High Bias):
  Train Accuracy: 0.7242
  Test Accuracy:  0.7153
  Accuracy Gap:   0.0088
```

Exact results depend on the supplied dataset.

---

# Running Tests

Run:

```bash
python3 -m pytest tests.py -v
```

Or:

```bash
python3 -m pytest tests.py -vv
```

The supplied tests verify that:

- `preprocess_data()` exists and is callable
- `cleaned_patient_data.csv` is generated
- the cleaned dataset contains no missing values
- the cleaned dataset contains no object columns
- `AtRisk` remains in the cleaned dataset
- `train_model()` exists and is callable
- `knn_k1_model.joblib` is generated
- `knn_k50_model.joblib` is generated
- `evaluate_model()` exists and is callable

---

# Key Learning

The central idea demonstrated by this project is:

```text
Small k
→ More complex decision boundary
→ Lower bias
→ Higher variance
→ Greater risk of overfitting

Large k
→ Smoother decision boundary
→ Higher bias
→ Lower variance
→ Greater risk of underfitting
```

The goal of model tuning is not simply to maximize training accuracy.

The goal is to choose a model complexity that performs well on **unseen data**.