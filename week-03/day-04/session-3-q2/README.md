# Product Defect Detection – Support Vector Classification with RBF Kernel

## Project Overview

This project builds a Support Vector Classification (SVC) machine learning pipeline to detect whether a manufactured product has a defect.

The pipeline:

- Cleans and preprocesses product inspection data.
- Handles missing numerical and categorical values.
- Applies one-hot encoding to categorical features.
- Splits the data into training and testing sets.
- Trains an SVC model using an RBF kernel.
- Evaluates the model using accuracy, F1-score, and a classification report.
- Saves the trained model and evaluation results to disk.

## Project Structure

```text
Project/
├── product_inspection.csv
├── main.py
├── tests.py
├── installation.txt
└── .gitignore
```

## Input Dataset

### `product_inspection.csv`

Training dataset containing historical product inspection records.

Columns:

- `ProductCategory`
- `ShiftType`
- `Temperature`
- `ProductionSpeed`
- `InspectionScore`
- `DefectFound`

### Target Variable

`DefectFound` is the target variable:

- `0` = No Defect
- `1` = Defect Found

## Dataset Details

| Column | Type | Description |
|---|---|---|
| `ProductCategory` | String | Product type such as Electronics, Textile, Automotive, or Pharma |
| `ShiftType` | String | Production shift: Morning, Afternoon, or Night |
| `Temperature` | Float | Machine temperature during production |
| `ProductionSpeed` | Float | Production speed in units per minute |
| `InspectionScore` | Float | Quality inspection score |
| `DefectFound` | Integer | Target label: 0 = No Defect, 1 = Defect Found |

The dataset contains approximately 3% missing values across feature columns. Missing values must be handled during preprocessing.

## Required Functions

### 1. `preprocess_data(dataframe)`

Preprocesses the raw inspection dataset.

The function must:

1. Identify numerical and categorical columns.
2. Fill missing numerical values using the median of each column.
3. Fill missing categorical values using the most frequent value (mode).
4. Apply one-hot encoding to categorical columns.
5. Use `drop_first=True` during one-hot encoding.
6. Return the cleaned and encoded DataFrame.

The returned DataFrame must contain no missing values and no object/string columns.

## 2. `train_model(cleaned_data)`

Trains the Support Vector Classification model.

The function must:

1. Separate the `DefectFound` target column from the feature columns.
2. Split the data into:
   - 80% training data
   - 20% testing data
3. Use `random_state=42` for the train/test split.
4. Initialize an SVC model with:
   - `kernel="rbf"`
   - `random_state=42`
5. Fit the model using the training features and target values.
6. Save the trained model using joblib as:

```text
svc_model.joblib
```

7. Return:

```text
model, features_test, target_test
```

Where:

- `model` = trained SVC model
- `features_test` = testing feature DataFrame
- `target_test` = testing target Series

## 3. `evaluate_model(model, features_test, target_test)`

Evaluates the trained SVC model.

The function must:

1. Generate predictions using the test features.
2. Calculate the accuracy score.
3. Calculate the F1-score.
4. Generate a classification report using the actual and predicted values.
5. Save the evaluation metrics to:

```text
evaluation_results.csv
```

6. Return:

```text
accuracy, f1, report
```

Where:

- `accuracy` = model accuracy as a float
- `f1` = F1-score as a float
- `report` = classification report as a string

## Main Execution

The main execution block must run only when `main.py` is executed directly:

```python
if __name__ == "__main__":
```

It should:

1. Load `product_inspection.csv`.
2. Preprocess the training data using `preprocess_data()`.
3. Save the cleaned dataset as:

```text
cleaned_inspection_data.csv
```

4. Train the SVC model using `train_model()`.
5. Save the trained model as:

```text
svc_model.joblib
```

6. Evaluate the model using `evaluate_model()`.
7. Save the evaluation results as:

```text
evaluation_results.csv
```

8. Print the accuracy, F1-score, and classification report to the console.

## Generated Files

After successful execution, the following files should be created:

| File | Description |
|---|---|
| `cleaned_inspection_data.csv` | Preprocessed inspection data with missing values handled |
| `svc_model.joblib` | Trained SVC model using an RBF kernel |
| `evaluation_results.csv` | Model evaluation metrics containing accuracy and F1-score |

## Handling Missing Values

Missing feature values must be handled before model training.

### Numerical Columns

Use the median of each numerical column:

```python
df[column] = df[column].fillna(df[column].median())
```

### Categorical Columns

Use the most frequent value:

```python
df[column] = df[column].fillna(df[column].mode().iloc[0])
```

The `DefectFound` target column does not contain missing values.

## Categorical Encoding

Categorical columns must be converted to numerical features using one-hot encoding:

```python
pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)
```

This allows the SVC model to process the categorical information numerically.

## Model Configuration

The required model is Support Vector Classification with an RBF kernel:

```python
SVC(
    kernel="rbf",
    random_state=42
)
```

The RBF kernel allows the classifier to model non-linear relationships between the inspection features and defect labels.

## Train/Test Split

The cleaned dataset must be split into training and testing sets using an 80/20 split:

```python
train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)
```

The test set is used only for evaluating the trained model.

## Evaluation Metrics

The model must produce:

### Accuracy

Measures the proportion of correctly classified products:

```text
Accuracy = Correct Predictions / Total Predictions
```

### F1-Score

The F1-score combines precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Classification Report

The classification report should contain:

- Precision
- Recall
- F1-score
- Support

for each class.

## Expected Output

Example console output:

```text
Cleaned data saved as 'cleaned_inspection_data.csv'
Model saved as 'svc_model.joblib'
Evaluation results saved to 'evaluation_results.csv'

Model Evaluation Results:
Accuracy : 0.8713
F1-Score : 0.8728

Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.87      0.87      1490
           1       0.87      0.88      0.87      1510

    accuracy                           0.87      3000
   macro avg       0.87      0.87      0.87      3000
weighted avg       0.87      0.87      0.87      3000
```

The exact evaluation values may vary depending on the dataset.

## Running the Project

Navigate to the project directory:

```bash
cd Project
```

Install the required dependencies:

```bash
pip install -r installation.txt
```

Run the main program:

```bash
python3 main.py
```

## Running Tests

Run the test suite with:

```bash
python3 -m pytest tests.py -v
```

or:

```bash
python3 -m pytest tests.py
```

## Test Coverage

The provided tests validate:

- `preprocess_data()` exists and is callable.
- Preprocessing removes all missing values.
- No object/string columns remain after preprocessing.
- `DefectFound` remains as the target column.
- `train_model()` exists and is callable.
- `svc_model.joblib` is created.
- Test features and target values are returned.
- `evaluate_model()` exists and is callable.
- Accuracy is between 0 and 1.
- F1-score is between 0 and 1.
- `evaluation_results.csv` is created.
- The classification report is returned as a string.

## Important Notes

- Do not remove the `.gitignore` file created outside the project directory.
- Missing values must be handled before training the SVC model.
- Categorical features must be one-hot encoded with `drop_first=True`.
- The target column is `DefectFound`.
- The model must use an RBF kernel.
- Use `random_state=42` wherever explicitly required.
- The dataset must be split into 80% training and 20% testing data.
- Evaluation must be performed on the test set.
- The project must generate all three required output files.
- The project should be submitted only after running the tests successfully.

## Execution Instructions

### Through the Project Interface

Click the **Run Test Case** button to execute and check the test case status.

### Through the Terminal

Navigate to the project directory:

```bash
cd Project
```

Install dependencies:

```bash
pip install -r installation.txt
```

Run the main program:

```bash
python3 main.py
```

Run the tests:

```bash
python3 -m pytest tests.py -v
```

## Submission

Make sure the project has been executed at least once before submitting.

The project should contain:

```text
product_inspection.csv
main.py
tests.py
installation.txt
```

and should generate:

```text
cleaned_inspection_data.csv
svc_model.joblib
evaluation_results.csv
```