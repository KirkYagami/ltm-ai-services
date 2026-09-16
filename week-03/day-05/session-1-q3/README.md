# Customer Segmentation Using K-Means Clustering

## Project Overview

This project builds an unsupervised machine learning pipeline for customer segmentation using K-Means clustering.

Customer records are preprocessed and grouped into three customer segments based on demographic, purchasing, and browsing-related features.

The pipeline performs:

- Missing value handling
- Categorical feature encoding
- Numerical feature standardization
- K-Means clustering
- Customer segment assignment
- Segment-level summary generation

---

## Project Structure

```text
Project/
├── ecommerce_customers.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_ecommerce_data.csv
├── scaler.pkl
├── kmeans_model.pkl
└── segmented_customers.csv
```

The last four files are generated when the program runs.

---

## Dataset

Input dataset:

```text
ecommerce_customers.csv
```

### Numerical Features

- `Age`
- `AnnualIncome`
- `SpendingScore`
- `PurchaseFrequency`
- `AvgOrderValue`

### Categorical Features

- `PreferredCategory`
- `DeviceType`

There is no target variable because K-Means is an unsupervised learning algorithm.

---

## Task 1: Data Preprocessing

Function:

```python
preprocess_data()
```

The function:

1. Loads `ecommerce_customers.csv`.
2. Automatically identifies numerical and categorical columns.
3. Replaces missing numerical values with their column mean.
4. Replaces missing categorical values with their column mode.
5. Saves the cleaned dataset as:

```text
cleaned_ecommerce_data.csv
```

6. Creates a copy of the cleaned original dataset.
7. One-hot encodes categorical features.
8. Standardizes numerical features using `StandardScaler`.
9. Saves the fitted scaler as:

```text
scaler.pkl
```

The function returns:

```python
features, original_dataset
```

`features` contains the encoded and standardized feature matrix used by K-Means.

`original_dataset` contains the cleaned data before encoding and scaling and is used later when assigning customer segment labels.

---

## Task 2: K-Means Model Training

Function:

```python
train_model(features)
```

The K-Means model is configured with:

```python
KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
```

The model groups customers into three clusters.

The trained model is saved as:

```text
kmeans_model.pkl
```

The function returns the trained K-Means model.

---

## Task 3: Customer Segment Assignment

Function:

```python
assign_segments(model, original_dataset)
```

Cluster labels are obtained from:

```python
model.labels_
```

A new column called:

```text
CustomerSegment
```

is added to the cleaned original dataset.

Possible segment labels are:

```text
0
1
2
```

The resulting dataset is saved as:

```text
segmented_customers.csv
```

The function also calculates the mean of each numerical feature grouped by `CustomerSegment`.

The segmented DataFrame is returned.

---

## Generated Files

| File | Description |
|---|---|
| `cleaned_ecommerce_data.csv` | Customer data after missing-value imputation |
| `scaler.pkl` | Fitted StandardScaler object |
| `kmeans_model.pkl` | Trained K-Means clustering model |
| `segmented_customers.csv` | Customer data with CustomerSegment labels |

---

## Installation

Install the required packages:

```bash
pip install -r installation.txt
```

Required dependencies include:

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

1. Clean the customer dataset.
2. Encode categorical features.
3. Standardize numerical features.
4. Train the K-Means model.
5. Assign customers to three segments.
6. Save the generated files.
7. Print the customer segmentation summary.

Example:

```text
Customer Segmentation Summary:

                  Age  AnnualIncome  SpendingScore  PurchaseFrequency  AvgOrderValue
CustomerSegment
0               39.76      60728.16          51.14               4.36          73.48
1               54.43     100190.25          29.61               1.80         202.06
2               25.19      31151.20          72.62               9.44          28.09
```

The exact values depend on the supplied dataset.

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

- `preprocess_data()` exists and is callable.
- Missing values are removed.
- `cleaned_ecommerce_data.csv` is created.
- `scaler.pkl` is created.
- `train_model()` exists and is callable.
- `kmeans_model.pkl` is created.
- `assign_segments()` exists and is callable.
- `segmented_customers.csv` is created.
- `CustomerSegment` is present.
- Exactly three customer segments are produced.
- The segmented dataset has the same number of rows as the original dataset.