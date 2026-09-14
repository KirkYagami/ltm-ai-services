# CUSTOMER SUBSCRIPTION PREDICTION USING LOGISTIC REGRESSION

## PROBLEM STATEMENT

This project focuses on implementing a complete binary classification pipeline to predict whether a customer is likely to subscribe to a service using Logistic Regression. You are working as a Data Scientist in a marketing analytics team, where understanding customer behavior and predicting subscription likelihood is critical for targeted marketing campaigns.

The project demonstrates a full end-to-end machine learning workflow using a single structured dataset, similar in complexity to standard datasets such as Iris or Titanic.

The system includes:
- Loading structured data from an Excel file
- Handling missing values in numerical and categorical columns
- Encoding categorical variables using ordinal encoding and one-hot encoding
- Splitting data into training and testing sets
- Training a Logistic Regression classification model
- Generating binary predictions on unseen data

The objective is to predict the binary outcome:
- `1`: Customer subscribed
- `0`: Customer did not subscribe

---

## FILE STRUCTURE

```text
Project/
├── main.py
├── customer_data.xlsx
├── requirements.txt
├── processed_data.xlsx (generated after cleaning)
└── tests.py
```

### FILE PURPOSES

#### main.py
Contains the full implementation of the customer subscription prediction pipeline, including data loading, preprocessing, model training, and prediction.

*Students are required to implement only this file.*

#### customer_data.xlsx
Training dataset containing customer demographic and account information.

#### requirements.txt
Specifies all required Python libraries and dependencies.

#### tests.py
Used internally for automated validation of the solution.

---

## DATASET DESCRIPTION

### Training Dataset Schema
**File:** `customer_data.xlsx`

| Column Name | Data Type | Description |
| --- | --- | --- |
| age | Numerical | Customer's age |
| job | Categorical (Non-Ordinal) | Customer's job type |
| marital_status | Categorical (Ordinal) | Relationship status |
| balance | Numerical | Average yearly account balance |
| contact | Categorical (Non-Ordinal) | Contact method |
| subscribed | Binary (Target) | 1 = subscribed, 0 = not subscribed |

### Ordinal Relationship
The `marital_status` column follows a strict ordinal hierarchy:
`single` < `married` < `divorced`

---

## MODULE EXPLANATION

**Module Name:** `main.py`

### FUNCTION DEFINITIONS

#### FUNCTION 1: load_dataset

##### Purpose
Loads the customer dataset from disk for further processing.

##### Function Signature
`load_dataset(file_path)`

##### Parameters
* `file_path` (string): Path to the Excel dataset file.

##### Function Logic
- Loads the dataset using `pandas.read_excel`
- Does not modify or preprocess the data

##### Return Type
`pandas.DataFrame`

---

#### FUNCTION 2: preprocess_data

##### Purpose
Cleans the dataset by handling missing values and encoding categorical features so that the data becomes suitable for Logistic Regression.

##### Function Signature
`preprocess_data(df)`

##### Missing Value Handling (Mandatory)
- **Numerical Columns (`age`, `balance`):** Fill with Mean
- **Categorical Columns (`job`, `contact`, `marital_status`):** Fill with Mode (Most Frequent Value)
- **After preprocessing:** `processed_df.isnull().sum().sum() == 0` (No missing values must remain)

##### Encoding Rules (Must Be Followed Exactly)
- **Ordinal Encoding (`marital_status`):** Use explicit mapping:
  ```json
  {
    "single": 0,
    "married": 1,
    "divorced": 2
  }
  ```
- **One-Hot Encoding (Non-Ordinal Features: `job`, `contact`):**
  Apply one-hot encoding with `drop_first=True` for `job` and `contact`. This prevents multicollinearity and ensures compatibility with Logistic Regression.

##### Return Value
`processed_df`: Fully cleaned and encoded pandas DataFrame.

---

#### FUNCTION 3: split_dataset

##### Purpose
Separates features and target variable and splits the dataset into training and testing sets.

##### Function Signature
`split_dataset(df)`

##### Feature Selection
- **Features (X):** All columns except `subscribed`
- **Target (y):** `subscribed`

##### Train-Test Split Configuration
- `test_size = 0.2`
- `random_state = 42`

##### Return Values (Exact Order)
`X_train`, `X_test`, `y_train`, `y_test`

---

#### FUNCTION 4: train_logistic_model

##### Purpose
Trains a Logistic Regression classification model using the training dataset.

##### Function Signature
`train_logistic_model(X_train, y_train)`

##### Requirements
- Use `sklearn.linear_model.LogisticRegression`
- Set `max_iter = 1000`
- Fit the model on training data
- Model must learn coefficients (`model.coef_`)

##### Return Type
Trained `LogisticRegression` model object

---

## MAIN EXECUTION FLOW

The main execution block in `main.py` must perform the following steps in sequence:
1. Load the dataset from `customer_data.xlsx`
2. Print a dataset preview to the console
3. Preprocess and encode the dataset
4. Save the processed dataset to disk (`processed_data.xlsx`)
5. Split the dataset into training and testing sets
6. Train the Logistic Regression model
7. Generate predictions on the test dataset
8. Print predictions to the console

---

## CONSOLE OUTPUT FORMAT

### Dataset Preview:
`<first few rows of the dataset>`

### Predictions Output
```text
Logistic Regression Predictions:
[0 1 0 0 1 ...]
```
*Predictions must be binary values only (0 or 1).*

---

## OUTPUT FILES

| File Name | Description |
| --- | --- |
| `processed_data.xlsx` | Fully cleaned and encoded dataset |

---

## IMPLEMENTATION INSTRUCTIONS

### Environment Setup
Navigate to the project directory:
```bash
cd Project
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Program
```bash
python3 main.py
```

### Running Validation
```bash
python3 -m pytest tests.py -v
```
