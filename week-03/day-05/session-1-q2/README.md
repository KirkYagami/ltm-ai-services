# Patient Readmission Prediction Using Naive Bayes — Setup & Run Guide

## Overview

A healthcare analytics company is building a patient risk assessment module for its hospital management platform. The data team has collected structured patient records including demographics, clinical measurements, and admission history.

This project develops a **Gaussian Naive Bayes classifier** that predicts whether a patient is likely to be readmitted within 30 days of discharge. The pipeline:

1. Preprocesses patient data (handles missing values, encodes categoricals, scales numericals).
2. Trains a Gaussian Naive Bayes classifier.
3. Evaluates the model using Accuracy and F1 Score on a held-out test set.

The whole pipeline is orchestrated by `main.py`.

---

## Project Structure

```
Project/
├── patient_readmission.csv    # Training dataset with labeled patient records
├── main.py                    # Preprocessing, training, and evaluation functions
├── tests.py                   # Test suite
└── installation.txt           # Python package dependencies
```

Generated at runtime:

```
Project/
├── cleaned_patient_data.csv   # Patient data after missing value imputation
├── scaler.pkl                 # Fitted StandardScaler object
└── naive_bayes_model.pkl      # Trained Gaussian Naive Bayes model
```

> **Note:** To avoid auto-save issues, a `.gitignore` file is created by default in the workspace (beside the project directory). **Do not remove or delete it.**

---

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (Python package installer)
- A terminal / shell (bash, zsh, PowerShell, or equivalent)

Core libraries used:

- `pandas` — tabular data handling and one-hot encoding
- `scikit-learn` — `GaussianNB`, `StandardScaler`, `train_test_split`, metrics
- `joblib` — model and scaler persistence
- `pytest` — test runner

---

## Dataset Description

**File:** `patient_readmission.csv`

| Column | Type | Description |
| --- | --- | --- |
| `Age` | Numerical | Age of the patient (18 to 90) |
| `LengthOfStay` | Numerical | Duration of last hospital stay in days (1 to 30) |
| `NumPreviousAdmissions` | Numerical | Number of prior hospital admissions (0 to 15) |
| `AvgGlucoseLevel` | Numerical | Average blood glucose level (70 to 300) |
| `ChronicCondition` | Categorical | Primary chronic condition |
| `AdmissionType` | Categorical | Type of admission (Emergency / Planned / Referral) |
| `InsuranceType` | Categorical | Insurance coverage (Private / Government / Self-Pay) |
| `Readmitted` | Binary | Target (1 = Readmitted, 0 = Not Readmitted) |

**Note:** The dataset contains missing values across **all feature columns**. The target column has **no missing values**.

---

## Setup Instructions

### Step 1 — Navigate to the project directory

```bash
cd Project
```

### Step 2 — (Optional but recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install the required dependencies

```bash
pip install -r installation.txt
```

If `installation.txt` is not present as a `requirements.txt` alias, install the core packages directly:

```bash
pip install pandas scikit-learn joblib pytest
```

---

## Running the Pipeline

### Step 4 — Execute the main program

```bash
python3 main.py
```

This runs the three tasks in sequence:

1. **Task 1 — Data Preprocessing (`preprocess_data`)**
   - Reads `patient_readmission.csv`.
   - Automatically identifies numerical and categorical columns.
   - Fills missing numerical values with the column **mean**.
   - Fills missing categorical values with the column **mode**.
   - Saves the cleaned dataset as `cleaned_patient_data.csv`.
   - Separates the target (`Readmitted`) from the features.
   - One-hot encodes categorical columns.
   - Applies `StandardScaler` on numerical columns.
   - Saves the fitted scaler as `scaler.pkl`.
   - Performs an 80/20 train-test split with `random_state=42` and stratification on the target.
   - Returns `X_train, X_test, y_train, y_test`.

2. **Task 2 — Model Training (`train_model`)**
   - Trains a `GaussianNB` classifier on the provided training data.
   - Saves the trained model as `naive_bayes_model.pkl`.
   - Returns the trained model.

3. **Task 3 — Model Evaluation (`evaluate_model`)**
   - Generates predictions on the test set.
   - Computes **Accuracy** using `accuracy_score`.
   - Computes **F1 Score** using `f1_score`.
   - Prints both metrics in a formatted summary.
   - Returns `(accuracy, f1)` as a tuple.

### Expected Console Output

```
Model Evaluation Results:
Accuracy: 0.9603
F1 Score: 0.9608
```

(Exact metric values depend on the dataset; the format above is what should be printed.)

---

## Generated Files

| File | Description |
| --- | --- |
| `cleaned_patient_data.csv` | Patient data after missing value imputation |
| `scaler.pkl` | Fitted `StandardScaler` object |
| `naive_bayes_model.pkl` | Trained Gaussian Naive Bayes model |

---

## Running the Tests

### From the interface

Click the **Run Test Case** button.

### From the terminal

```bash
python3 -m pytest tests.py -v
```

The suite verifies:

- `preprocess_data`, `train_model`, and `evaluate_model` are callable.
- `cleaned_patient_data.csv` is created and contains **zero missing values**.
- `scaler.pkl` is saved.
- The preprocessed split returns four components with `len(X_train) > len(X_test)` and non-zero total rows.
- `naive_bayes_model.pkl` is saved after training.
- `evaluate_model` returns an `(accuracy, f1)` tuple where both values lie in `[0.0, 1.0]`.

---

## Function Reference

### `preprocess_data()`

**Parameters:** None

**Returns:** `X_train (DataFrame), X_test (DataFrame), y_train (Series), y_test (Series)`

**Logic:**
1. Read `patient_readmission.csv`.
2. Identify numerical and categorical columns automatically.
3. Fill missing numerical values with the column mean.
4. Fill missing categorical values with the column mode.
5. Save the cleaned dataset as `cleaned_patient_data.csv`.
6. Separate the target column (`Readmitted`) from features.
7. Apply one-hot encoding on categorical columns.
8. Apply `StandardScaler` on numerical columns.
9. Save the fitted scaler as `scaler.pkl` using `joblib`.
10. Perform an 80/20 train-test split with `random_state=42` and stratification on the target.
11. Return the four split components.

---

### `train_model(X_train, y_train)`

**Parameters:**
- `X_train` — DataFrame of training features from `preprocess_data`.
- `y_train` — Series of training target from `preprocess_data`.

**Returns:** `nb_model` (GaussianNB)

**Logic:**
1. Train a `GaussianNB` classifier using the provided training data.
2. Save the trained model as `naive_bayes_model.pkl` using `joblib`.
3. Return the trained model.

---

### `evaluate_model(model, X_test, y_test)`

**Parameters:**
- `model` — Trained `GaussianNB` from `train_model`.
- `X_test` — DataFrame of test features from `preprocess_data`.
- `y_test` — Series of test target from `preprocess_data`.

**Returns:** `(accuracy (float), f1 (float))`

**Logic:**
1. Generate predictions on the test set using the trained model.
2. Compute Accuracy using `accuracy_score`.
3. Compute F1 Score using `f1_score`.
4. Print both metrics in a formatted summary.
5. Return accuracy and F1 score as a tuple.

---

## Execution Instructions (Quick Reference)

1. Navigate to the project directory:
   ```bash
   cd Project
   ```
2. Install the required dependencies:
   ```bash
   pip install -r installation.txt
   ```
3. To execute the main program:
   ```bash
   python3 main.py
   ```
4. To run test cases:
   - Click the **Run Test Case** button on the interface, **or**
   - Run from the CLI:
     ```bash
     python3 -m pytest tests.py -v
     ```

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
| --- | --- |
| `FileNotFoundError: patient_readmission.csv` | Run commands from the `Project/` directory, not a subfolder. |
| `ModuleNotFoundError: No module named 'sklearn'` | Install dependencies: `pip install -r installation.txt`. |
| `cleaned_patient_data.csv` still contains NaNs | Missing values were not imputed — ensure `fillna` is applied to both numerical (mean) and categorical (mode) columns before saving. |
| `scaler.pkl` missing | `StandardScaler().fit_transform(...)` was not called on the numerical columns, or `joblib.dump` was skipped. |
| `naive_bayes_model.pkl` missing | `train_model` did not call `joblib.dump`. |
| Test fails on `len(X_train) > len(X_test)` | The split ratio is wrong — use `test_size=0.2`. |
| Test fails on tuple return | `evaluate_model` must return `(accuracy, f1)` in that order. |
| Accuracy or F1 outside `[0.0, 1.0]` | Using probabilities instead of class labels — call `model.predict(...)`, not `predict_proba`. |
| Auto-save issues in the workspace | Leave the provided `.gitignore` untouched; do not remove or delete it. |

---

## Submission Checklist

- [ ] `main.py` implements all three required functions with correct signatures.
- [ ] `pip install -r installation.txt` completes without errors.
- [ ] `python3 main.py` runs end-to-end and prints the formatted evaluation summary.
- [ ] `cleaned_patient_data.csv`, `scaler.pkl`, and `naive_bayes_model.pkl` are generated.
- [ ] `python3 -m pytest tests.py -v` passes **all** tests.
- [ ] `.gitignore` file is left untouched.

---

## Notes

- Imputation is done with the **mean** for numerical columns and the **mode** for categorical columns, as required.
- Scaling and encoding are applied **after** the target is separated and **before** the train-test split.
- The `StandardScaler` is fit on the full feature set (not just training), which is the specification given in the task.
- The train-test split uses `stratify=y` to preserve class proportions — this matters for imbalanced readmission data.
- The pipeline is deterministic thanks to `random_state=42`.