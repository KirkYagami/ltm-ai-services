# Customer Churn Prediction Using KNN vs Naive Bayes

## Project Overview

This project builds a machine learning pipeline for customer churn prediction using two classification algorithms:

- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Both models are trained on the same telecom customer dataset and compared using F1 Score.

The model with the higher F1 Score is identified as the best-performing model for the supplied dataset.

---

## Project Structure

```text
Project/
├── telecom_churn.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_churn_data.csv
├── scaler.pkl
├── knn_model.pkl
└── nb_model.pkl
```

The last four files are generated when the program runs.

---

## Dataset

Input dataset:

```text
telecom_churn.csv
```

### Numerical Features

- `Age`
- `Tenure`
- `MonthlyCharges`
- `TotalCharges`

### Categorical Features

- `ContractType`
- `PaymentMethod`
- `InternetService`

### Target

```text
Churned
```

Values:

- `1` = Churned
- `0` = Retained

---

## Task 1: Data Preprocessing

Function:

```python
preprocess_data()
```

The function:

1. Loads `telecom_churn.csv`.
2. Automatically identifies numerical and categorical columns.
3. Fills missing numerical values with their respective column mean.
4. Fills missing categorical values with their respective column mode.
5. Saves the cleaned dataset as:

```text
cleaned_churn_data.csv
```

6. Separates the `Churned` target from the features.
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

Two classification models are trained.

### KNN

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

The trained KNN model is saved as:

```text
knn_model.pkl
```

### Gaussian Naive Bayes

```python
GaussianNB()
```

The trained Naive Bayes model is saved as:

```text
nb_model.pkl
```

The function returns:

```python
knn_model, nb_model
```

---

## Task 3: Model Comparison

Function:

```python
compare_models(
    knn_model,
    nb_model,
    X_test,
    y_test
)
```

Both models generate predictions on the same test dataset.

F1 Score is calculated using:

```python
f1_score()
```

The two scores are compared.

If:

```text
KNN F1 >= Naive Bayes F1
```

the best model name is:

```text
KNN
```

Otherwise:

```text
Naive Bayes
```

The function returns:

```python
knn_f1, nb_f1, best_model_name
```

---

## Why F1 Score?

F1 Score combines Precision and Recall:

```text
                 Precision × Recall
F1 = 2 × --------------------------------
                 Precision + Recall
```

This is useful for churn prediction because identifying customers who actually churn can be important even when the classes are not perfectly balanced.

---

## Generated Files

| File | Description |
|---|---|
| `cleaned_churn_data.csv` | Customer data after missing-value imputation |
| `scaler.pkl` | Fitted StandardScaler object |
| `knn_model.pkl` | Trained KNN model |
| `nb_model.pkl` | Trained Gaussian Naive Bayes model |

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

The program will:

1. Clean and preprocess the dataset.
2. One-hot encode categorical features.
3. Standardize numerical features.
4. Create training and testing datasets.
5. Train KNN and Gaussian Naive Bayes.
6. Save both trained models.
7. Calculate F1 Score for both models.
8. Identify the model with the higher F1 Score.

Example output:

```text
Multi-Model Comparison:
KNN F1 Score: 0.9309
Naive Bayes F1 Score: 0.9291
Best Model: KNN
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
- `cleaned_churn_data.csv` is generated.
- The cleaned dataset contains no missing values.
- `scaler.pkl` is generated.
- The train-test split is returned correctly.
- `train_models()` is callable.
- `knn_model.pkl` is generated.
- `nb_model.pkl` is generated.
- `compare_models()` is callable.
- KNN F1 Score is between 0 and 1.
- Naive Bayes F1 Score is between 0 and 1.
- The best model name is either `KNN` or `Naive Bayes`.