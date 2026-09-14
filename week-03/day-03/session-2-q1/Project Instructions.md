# SALARY PREDICTION USING LINEAR REGRESSION

## PROBLEM STATEMENT

This project focuses on building a complete end-to-end salary prediction system using Linear Regression on a structured dataset. You are working as a Data Scientist in a recruitment analytics firm, where your task is to analyze historical candidate data and predict annual salaries for new candidates based on their professional background.

The project demonstrates the full machine learning workflow using tabular data, including:
- Loading datasets from CSV files
- Handling missing values in numerical and categorical data
- Encoding categorical variables for machine learning compatibility
- Splitting data into training and testing sets
- Training a Linear Regression model
- Predicting salaries for unseen candidates
- Saving processed datasets and prediction outputs to disk

The goal is to predict a candidate's annual salary using the following inputs:
- Years of professional experience
- Education level

This project is intentionally designed to mirror real-world machine learning pipelines using libraries such as pandas and scikit-learn.

---

## FILE STRUCTURE

```text
Project/
├── main.py
├── salary_data.csv
├── predictions.csv
├── predicted_salaries.csv
├── requirements.txt
└── tests.py
```

### FILE PURPOSES

#### main.py
Contains the complete implementation of the machine learning pipeline, including data loading, preprocessing, model training, and salary prediction.

*You are required to implement only this file.*

#### salary_data.csv
Training dataset containing historical candidate information including salary values.

#### predictions.csv
Dataset containing new candidate information for which salary predictions must be generated.

#### requirements.txt
Lists all required Python dependencies.

---

## DATASET DESCRIPTION

### Training Dataset Schema
**File:** `salary_data.csv`

| Column Name | Data Type | Description |
| --- | --- | --- |
| YearsExperience | Numeric | Total years of professional work experience |
| EducationLevel | Categorical | Education level of candidate (Bachelor, Master, PhD) |
| Salary | Numeric | Annual salary in USD (target variable) |

### Prediction Dataset Schema
**File:** `predictions.csv`

| Column Name | Data Type | Description |
| --- | --- | --- |
| YearsExperience | Numeric | Total years of professional work experience |
| EducationLevel | Categorical | Education level (Bachelor, Master, PhD) |

*This dataset does not contain Salary and is used only for prediction.*

---

## MODULE EXPLANATION

**Module Name:** `main.py`

### FUNCTION DEFINITIONS

#### FUNCTION 1: load_dataset

##### Purpose
Loads a dataset from disk into memory for further processing.

##### Function Signature
`load_dataset(file_path)`

##### Parameters
* `file_path` (string): Path to the CSV file to be loaded.

##### Function Logic
- Reads the CSV file using `pandas.read_csv`
- Does not modify or preprocess the data

##### Return Type
`pandas.DataFrame`

---

#### FUNCTION 2: preprocess_data

##### Purpose
Cleans the dataset by handling missing values and encoding categorical features so the data becomes suitable for machine learning.

##### Function Signature
`preprocess_data(df, encoder=None, fit_encoder=True)`

##### Parameters
* `df` (`pandas.DataFrame`): Input dataset to preprocess.
* `encoder` (`LabelEncoder` or `None`): Existing encoder to reuse during prediction.
* `fit_encoder` (`boolean`):
  * `True`: Fit a new encoder
  * `False`: Reuse the provided encoder

##### PREPROCESSING RULES (MANDATORY)

###### Missing Value Handling
- **YearsExperience:** Filled using the mean of the column.
- **EducationLevel:** Filled using the most frequent value (mode).
- **After preprocessing:** Total missing values must be zero.

###### CATEGORICAL ENCODING
- Encode `EducationLevel` using `LabelEncoder`.
- Mapping is learned during training and reused during prediction.
- No new categories should be fitted during prediction phase.

##### Return Values
Must return exactly TWO values:
1. `processed_df`: Fully cleaned and encoded DataFrame.
2. `encoder`: Fitted `LabelEncoder` instance.

*Returning the encoder is mandatory, as it is reused for prediction data.*

---

#### FUNCTION 3: split_data

##### Purpose
Separates features and target variable and performs train-test split.

##### Function Signature
`split_data(df)`

##### Feature Selection
- **Input Features (X):** `YearsExperience`, `EducationLevel`
- **Target Variable (y):** `Salary`

##### Splitting Configuration
- `test_size = 0.2`
- `random_state = 42`

##### Return Values
Exactly in this order: `X_train`, `X_test`, `y_train`, `y_test`

---

#### FUNCTION 4: train_model

##### Purpose
Trains a Linear Regression model on the training data.

##### Function Signature
`train_model(X_train, y_train)`

##### Requirements
- Use `sklearn.linear_model.LinearRegression`
- Fit the model using training data
- Model must learn two coefficients:
  - One for `YearsExperience`
  - One for `EducationLevel`

##### Return Type
Trained `LinearRegression` model object

---

#### FUNCTION 5: predict_new_data

##### Purpose
Generates salary predictions for new candidates and saves results to disk.

##### Function Signature
`predict_new_data(model, encoder, input_file, output_file)`

##### Function Logic
- Load `predictions.csv`
- Apply same preprocessing rules as training data
- Reuse the same `LabelEncoder`
- Predict salaries using trained model
- Add a new column: `PredictedSalary`
- Save results to output CSV file

##### Output File
- **File name:** `predicted_salaries.csv`
- **Must include:** `YearsExperience`, `EducationLevel` (encoded), `PredictedSalary`

---

## CONSOLE OUTPUT FORMAT

The program must print the following header before displaying predictions:

```text
Predicted Salaries:
```

Followed by the full prediction DataFrame.

---

## OUTPUT FILES

| File Name | Description |
| --- | --- |
| `predicted_salaries.csv` | Salary predictions for new candidates |

### OUTPUT EXECUTION SCREENSHOT

Just for column sample:

```text
Predicted Salaries:
   YearsExperience  EducationLevel  PredictedSalary
0              3.0               1     80547.981322
...
```

---

## IMPLEMENTATION INSTRUCTIONS

### Environment Setup
Navigate to project directory:
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
