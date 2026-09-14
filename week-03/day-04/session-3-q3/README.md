# Student Exam Outcome Prediction – Support Vector Classification with Polynomial Kernel

## Problem Statement

An online learning platform tracks student engagement and performance data across multiple course categories and study modes. The academic analytics team wants to build a machine learning model that predicts whether a student will pass or fail an exam based on study hours, attendance rate, assignment scores, and course details.

The project requires building a Support Vector Classification pipeline using a **polynomial kernel**.

## Objectives

- Clean and encode raw student performance data.
- Handle missing numerical and categorical values.
- Train an SVC model using a polynomial kernel.
- Generate exam outcome predictions for new students.
- Save the trained model and prediction results to files.

## File Structure

```text
Project/
├── student_performance.csv
├── new_students.csv
├── main.py
├── tests.py
├── installation.txt
└── PROJECT_INSTRUCTIONS.md
```

## Dataset

### Training Dataset

`student_performance.csv`

| Column | Type | Description |
|---|---|---|
| CourseCategory | String | Course type: Science / Commerce / Arts / Engineering |
| StudyMode | String | Mode of study: Online / Offline / Hybrid |
| HoursStudied | Float | Weekly study hours |
| AttendanceRate | Float | Class attendance percentage |
| AssignmentScore | Float | Average assignment score |
| ExamResult | Integer | Target: 0 = Fail, 1 = Pass |

The dataset contains approximately **3% missing values** across all columns except `ExamResult`.

Missing values must be handled during preprocessing.

### Prediction Dataset

`new_students.csv`

Contains the same feature columns as the training dataset, except `ExamResult`.

The dataset is used to generate exam outcome predictions.

## Required Functions

### 1. `preprocess_data`

```python
preprocess_data(dataframe)
```

Implement the following:

- Identify numerical and categorical columns.
- Fill missing numerical values using the median of each column.
- Fill missing categorical values using the most frequent value (mode).
- Apply one-hot encoding to categorical columns.
- Use `drop_first=True` for one-hot encoding.
- Return the cleaned and encoded DataFrame.

The function must work with both training data containing `ExamResult` and prediction data without `ExamResult`.

### 2. `train_model`

```python
train_model(cleaned_data)
```

Implement the following:

- Separate `ExamResult` from the feature columns.
- Store the feature column names in a list called `training_columns`.
- Initialize an `SVC` model using:
  - `kernel="poly"`
  - `random_state=42`
- Train the model using the complete cleaned training dataset.
- Save the trained model as:

```text
svc_model.joblib
```

Return:

```text
model, training_columns
```

### 3. `predict_result`

```python
predict_result(model, prediction_data, training_columns)
```

Implement the following:

- Preserve the original prediction DataFrame.
- Preprocess the prediction data using `preprocess_data`.
- Align prediction columns with the training columns.
- Add any missing training columns with a value of `0`.
- Reorder prediction columns to exactly match `training_columns`.
- Generate predictions using the trained model.
- Add the predictions as a new column named:

```text
PredictedResult
```

- Save the resulting DataFrame as:

```text
predicted_students.csv
```

- Return the resulting DataFrame.

## Main Execution Block

Use:

```python
if __name__ == "__main__":
```

The main block must:

1. Load `student_performance.csv`.
2. Load `new_students.csv`.
3. Call `preprocess_data()` on the training dataset.
4. Save the cleaned training data as:

```text
cleaned_student_data.csv
```

5. Call `train_model()` using the cleaned training data.
6. Call `predict_result()` using:
   - the trained model,
   - prediction data,
   - training columns.
7. Print sample predictions to the console.

## Generated Files

The program should generate:

| File | Description |
|---|---|
| `cleaned_student_data.csv` | Preprocessed training data with missing values handled and categorical columns encoded |
| `svc_model.joblib` | Trained SVC model using a polynomial kernel |
| `predicted_students.csv` | New student records with predicted exam outcomes |

## Expected Console Output

The program should display messages similar to:

```text
Cleaned data saved as 'cleaned_student_data.csv'
Model saved as 'svc_model.joblib'
Predictions saved to 'predicted_students.csv'

Sample Predictions:
...
```

## Dependencies

Install the required packages using:

```bash
pip install -r installation.txt
```

The project requires packages including:

```text
pandas
scikit-learn
joblib
pytest
```

## Running the Project

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

## Running Tests

Run all tests with:

```bash
python3 -m pytest tests.py -v
```

Alternatively:

```bash
python3 -m pytest -v
```

## Test Requirements

The implementation should pass tests verifying:

- `preprocess_data` exists and is callable.
- Preprocessed data contains no missing values.
- Preprocessed data contains no object/string columns.
- `ExamResult` remains present after preprocessing.
- `train_model` exists and is callable.
- `svc_model.joblib` is created.
- `train_model` returns non-empty training columns.
- `predict_result` exists and is callable.
- `predicted_students.csv` is created.
- The prediction DataFrame contains `PredictedResult`.
- The prediction output has the same number of rows as `new_students.csv`.

## Important Implementation Notes

### Missing Values

Numerical columns must use their column median:

```python
df[column] = df[column].fillna(df[column].median())
```

Categorical columns must use their mode:

```python
df[column] = df[column].fillna(df[column].mode().iloc[0])
```

### One-Hot Encoding

Categorical features should be encoded using:

```python
pd.get_dummies(df, columns=categorical_columns, drop_first=True)
```

### Feature Alignment

The prediction dataset can produce a different set of one-hot encoded columns from the training dataset. Therefore, prediction features must be aligned using the training columns:

```python
processed_data = processed_data.reindex(
    columns=training_columns,
    fill_value=0
)
```

This ensures that the model receives features in exactly the same order and structure used during training.

### Model

Use:

```python
from sklearn.svm import SVC

model = SVC(
    kernel="poly",
    random_state=42
)
```

Save it using:

```python
import joblib

joblib.dump(model, "svc_model.joblib")
```

## `.gitignore`

A `.gitignore` file is expected to exist in the workspace, outside the project directory.

**Do not remove or delete the `.gitignore` file.**

## Completion Checklist

Before submitting the project, verify:

- [ ] `main.py` exists.
- [ ] `tests.py` exists.
- [ ] `student_performance.csv` exists.
- [ ] `new_students.csv` exists.
- [ ] `installation.txt` exists.
- [ ] `preprocess_data()` is implemented.
- [ ] `train_model()` is implemented.
- [ ] `predict_result()` is implemented.
- [ ] Missing values are handled.
- [ ] Categorical columns are one-hot encoded with `drop_first=True`.
- [ ] SVC uses a polynomial kernel.
- [ ] `svc_model.joblib` is generated.
- [ ] `cleaned_student_data.csv` is generated.
- [ ] `predicted_students.csv` is generated.
- [ ] Prediction features are aligned with training columns.
- [ ] Tests pass successfully.

## Submission

Run the complete program at least once:

```bash
python3 main.py
```

Then verify the generated files and run:

```bash
python3 -m pytest tests.py -v
```

The project should be executed successfully and all required test cases should pass before submission.