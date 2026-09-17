# Student Performance — Feature Engineering and Selection

## Objective

Prepare raw student performance data for predictive modeling and select the most relevant features based on their correlation with `FinalGrade`.

The project demonstrates:

- Missing-value handling
- Numerical and categorical preprocessing
- One-hot encoding
- Feature-target correlation
- Absolute correlation
- Correlation-based feature ranking
- Top-K feature selection
- Dimensionality reduction

---

# Dataset

Input file:

```text
student_performance.csv
```

The target variable is:

```text
FinalGrade
```

## Features

| Feature | Type | Description |
|---|---|---|
| StudyHours | Float | Average daily study hours |
| Attendance | Float | Attendance percentage |
| ParentEducation | Integer | Parent education level |
| FamilyIncome | Float | Annual family income |
| TutorSessions | Integer | Number of tutoring sessions |
| SleepHours | Float | Average daily sleep |
| ScreenTime | Float | Daily screen time |
| PhysicalActivity | Float | Daily physical activity |
| PreviousScore | Float | Previous examination score |
| Gender | Categorical | Male/Female |
| FinalGrade | Float | Prediction target |

---

# Task 1 — `preprocess_data`

## Function Signature

```python
preprocess_data(filepath)
```

The function must:

1. Load the CSV using Pandas.
2. Record missing-value counts before imputation.
3. Identify numerical columns.
4. Identify categorical columns.
5. Replace missing numerical values using the median.
6. Replace missing categorical values using the mode.
7. Separate `FinalGrade` from the feature matrix.
8. One-hot encode categorical variables using `drop_first=True`.
9. Ensure the returned features contain no object columns.

Return:

```python
features, target, missing_before
```

where:

```text
features
    Cleaned and encoded feature DataFrame

target
    FinalGrade Series

missing_before
    Original missing-value counts
```

---

# Numerical Imputation

Missing numerical values are replaced using the median:

```python
data[column].fillna(data[column].median())
```

Example:

```text
StudyHours

2
4
NaN
6
8
```

Median:

```text
6
```

Result:

```text
2
4
6
6
8
```

---

# Categorical Imputation

Categorical missing values are replaced with the mode.

```python
data[column].mode()[0]
```

The mode is the most frequently occurring category.

---

# One-Hot Encoding

Categorical features are converted to numerical features using:

```python
pd.get_dummies(
    features,
    drop_first=True
)
```

For example:

```text
Gender

Female
Male
Male
Female
```

becomes:

```text
Gender_Male

0
1
1
0
```

This ensures that the feature matrix contains only numerical values.

---

# Task 2 — `select_features`

## Function Signature

```python
select_features(features, target, top_k=5)
```

The function calculates the correlation between every feature and `FinalGrade`.

Use:

```python
features.corrwith(target)
```

Then take absolute values:

```python
features.corrwith(target).abs()
```

Sort them from highest to lowest:

```python
correlations.sort_values(
    ascending=False
)
```

Select the first `top_k` feature names.

For:

```python
top_k=5
```

exactly five features must be returned.

Return:

```python
selected_features, correlations
```

---

# Main Execution

The main program should:

```text
Load Dataset
     ↓
Preprocess Data
     ↓
Display Missing Values
     ↓
Calculate Correlations
     ↓
Rank Features
     ↓
Select Top 5
     ↓
Display Dimensionality Reduction
```

Run:

```bash
python main.py
```

Run tests:

```bash
pytest tests.py -v
```

---

# Expected Project Structure

```text
Project/
│
├── student_performance.csv
├── main.py
├── tests.py
├── PROJECT_INSTRUCTIONS.md
├── README.md
└── installation.txt
```

For this exercise the primary dependency is:

```text
pandas
```

For testing:

```text
pytest
```