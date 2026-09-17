# Student Performance Feature Engineering & Selection

This project demonstrates how raw data can be transformed into a model-ready feature matrix and then reduced by selecting features having the strongest relationships with the target.

The workflow is:

```text
Raw Dataset
     ↓
Missing-Value Handling
     ↓
Categorical Encoding
     ↓
Numerical Feature Matrix
     ↓
Feature-Target Correlation
     ↓
Rank Features
     ↓
Select Top-K Features
```

---

# 1. What is Feature Engineering?

**Feature engineering** is the process of preparing, transforming, or creating input variables so they can be effectively used by machine-learning algorithms.

Raw data:

```text
StudyHours     5.0
Attendance     77.1
Gender         Male
FamilyIncome   55835
```

is not necessarily immediately model-ready.

For example:

- Missing values may exist.
- Categorical values such as `Male` are strings.
- Some variables may contribute little useful information.
- The dataset may contain more dimensions than necessary.

Feature engineering transforms this into a usable numerical representation.

---

# 2. Features and Target

The dataset contains ten input features and one target.

Conceptually:

```text
X = Features

StudyHours
Attendance
ParentEducation
FamilyIncome
TutorSessions
SleepHours
ScreenTime
PhysicalActivity
PreviousScore
Gender
```

The prediction target is:

```text
y = FinalGrade
```

Therefore:

```text
X ───────────────→ ML Model ───────────────→ y
Student Features                            FinalGrade
```

The target must not remain inside `X`.

Otherwise, the model could accidentally receive the answer it is supposed to predict.

This is known as **target leakage**.

---

# 3. Missing Values

The dataset contains missing values in features such as:

```text
StudyHours
Attendance
FamilyIncome
SleepHours
```

Dropping every row containing a missing value could unnecessarily discard useful information.

Instead, the project performs **imputation**.

---

# 4. Median Imputation

Numerical missing values are replaced using the median.

Example:

```text
FamilyIncome

20000
30000
40000
50000
1000000
```

Median:

```text
40000
```

The median is often useful because it is relatively resistant to extreme values.

Compare:

```text
20,000
30,000
40,000
50,000
1,000,000
```

The very large final value can strongly influence the mean, whereas the median remains the middle observation.

The project therefore uses:

```python
data[column].fillna(
    data[column].median()
)
```

---

# 5. Mode Imputation

Categorical features cannot meaningfully use a numerical median.

Instead, missing categorical observations are replaced by the **mode**.

The mode is simply the most common value.

Example:

```text
Gender

Male
Female
Male
NaN
Male
```

Mode:

```text
Male
```

Therefore:

```text
NaN → Male
```

In Pandas:

```python
data[column].mode()[0]
```

returns the first mode.

---

# 6. One-Hot Encoding

Machine-learning algorithms generally require numerical features.

Suppose:

```text
Gender

Male
Female
Male
Female
```

We should not arbitrarily encode this as:

```text
Male   = 1
Female = 2
```

because numerical values can imply a quantitative ordering that does not exist.

Instead, one-hot encoding creates indicator variables.

With:

```python
pd.get_dummies(
    features,
    drop_first=True
)
```

we might obtain:

```text
Gender_Male

1
0
1
0
```

---

# 7. Why `drop_first=True`?

Suppose a categorical feature has two possible values:

```text
Male
Female
```

Without dropping a category:

```text
Gender_Female    Gender_Male

1                0
0                1
```

One column is completely determined by the other:

```text
Gender_Male = 1 - Gender_Female
```

Therefore one category can be used as the reference category.

With:

```python
drop_first=True
```

only:

```text
Gender_Male
```

needs to remain.

Interpretation:

```text
Gender_Male = 1 → Male
Gender_Male = 0 → Reference category
```

---

# 8. What is Feature Selection?

After preprocessing, we have a completely numerical feature matrix.

However, not every feature necessarily contributes equally to predicting `FinalGrade`.

**Feature selection** means choosing a subset of the existing features considered most relevant to the prediction problem.

Example:

```text
10 Features
     ↓
Evaluate Relevance
     ↓
Rank Features
     ↓
Keep Top 5
     ↓
5 Features
```

Unlike feature extraction methods such as PCA, feature selection retains actual original feature columns.

---

# 9. Why Perform Feature Selection?

Feature selection can:

- Reduce dimensionality
- Reduce unnecessary input variables
- Decrease computational requirements
- Simplify models
- Improve interpretability
- Sometimes improve generalization

However, removing features is not automatically beneficial.

A feature that looks weak individually may still be useful when combined with other features.

---

# 10. Correlation

This project uses **Pearson correlation** to measure the linear relationship between each feature and `FinalGrade`.

Correlation is commonly represented by:

```text
r
```

and ranges from:

```text
-1 ≤ r ≤ +1
```

Interpretation:

```text
r ≈ +1
Strong positive linear relationship

r ≈ 0
Weak/no linear relationship

r ≈ -1
Strong negative linear relationship
```

---

# 11. Positive Correlation

Suppose:

```text
StudyHours ↑
FinalGrade ↑
```

Then we have a positive relationship.

Example:

```text
StudyHours     FinalGrade

2              50
4              60
6              70
8              80
```

Conceptually:

```text
Grade
 ^
 |             *
 |          *
 |       *
 |    *
 +------------------> StudyHours
```

The correlation approaches:

```text
+1
```

as the points increasingly follow a perfect upward straight line.

---

# 12. Negative Correlation

Suppose:

```text
ScreenTime ↑
FinalGrade ↓
```

A possible dataset could look like:

```text
ScreenTime     Grade

2              90
4              80
6              70
8              60
```

Conceptually:

```text
Grade
 ^
 | *
 |    *
 |       *
 |          *
 +------------------> ScreenTime
```

This produces negative correlation.

The closer the linear relationship is to perfectly downward, the closer:

```text
r → -1
```

---

# 13. Why Use Absolute Correlation?

For feature selection, this project cares about the **strength** of the relationship rather than its direction.

Suppose:

```text
Feature A correlation = +0.70
Feature B correlation = -0.80
```

Feature B actually has the stronger linear relationship.

If we simply sorted normally:

```text
+0.70 > -0.80
```

we could incorrectly rank Feature A higher.

Therefore:

```python
features.corrwith(target).abs()
```

converts them to:

```text
Feature A = 0.70
Feature B = 0.80
```

Now Feature B correctly ranks higher based on relationship strength.

---

# 14. Calculating Correlation in Pandas

The project uses:

```python
correlations = features.corrwith(target)
```

Conceptually Pandas performs:

```text
StudyHours ───────────┐
Attendance ───────────┤
PreviousScore ────────┤
ScreenTime ───────────┼──→ Correlation with FinalGrade
SleepHours ───────────┤
FamilyIncome ─────────┤
...                   │
                      ▼
                   FinalGrade
```

Then:

```python
.abs()
```

keeps the magnitude.

Finally:

```python
.sort_values(ascending=False)
```

ranks strongest to weakest.

---

# 15. Correlation-Based Feature Selection

The complete algorithm is remarkably small:

```python
correlations = (
    features
    .corrwith(target)
    .abs()
    .sort_values(ascending=False)
)
```

Then:

```python
top_features = correlations.head(5).index
```

provides the names of the five strongest features.

Finally:

```python
selected_features = features[top_features]
```

creates the reduced feature matrix.

---

# 16. Example from the Dataset

The expected output shows approximately:

```text
StudyHours          0.5972
PreviousScore       0.4095
Attendance          0.2847
ScreenTime          0.2578
ParentEducation     0.1700
TutorSessions       0.1542
SleepHours          0.1377
PhysicalActivity    0.0362
Gender_Male         0.0077
FamilyIncome        0.0031
```

Because we request:

```python
top_k=5
```

the selected features become:

```text
StudyHours
PreviousScore
Attendance
ScreenTime
ParentEducation
```

The feature matrix therefore changes from:

```text
(10000, 10)
```

to:

```text
(10000, 5)
```

The number of rows remains unchanged.

Only the number of columns decreases.

---

# 17. What Does `0.5972` Mean?

For:

```text
StudyHours → 0.5972
```

this means there is a moderately strong linear association between `StudyHours` and `FinalGrade` in this dataset.

It does **not** mean:

```text
StudyHours causes 59.72% of the grade
```

and it does not mean:

```text
StudyHours explains exactly 59.72% of performance.
```

Correlation measures association, not causation.

---

# 18. Correlation Does Not Imply Causation

Suppose we observe:

```text
StudyHours ↑
FinalGrade ↑
```

We can say:

> The variables are positively associated in this dataset.

We cannot conclude from correlation alone:

> Increasing study hours directly causes the observed grade increase.

Other variables could be involved.

For example:

```text
Motivation
    │
    ├────────→ StudyHours
    │
    └────────→ FinalGrade
```

This illustrates why correlation and causation are different concepts.

---

# 19. Limitation — Correlation Detects Linear Relationships

This is one of the most important limitations of this feature-selection method.

Suppose:

```math
y = x²
```

The relationship is extremely strong.

But:

```text
          *       *
            *   *
              *
            *   *
          *       *
```

Depending on the distribution of `x`, Pearson correlation between `x` and `y` can be near zero because the relationship is nonlinear and symmetric.

Therefore:

```text
Low Correlation
       ≠
No Predictive Information
```

It may simply mean:

```text
No strong LINEAR relationship
```

---

# 20. Filter-Based Feature Selection

The technique used here is called a **filter method**.

```text
             FEATURE SELECTION

                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
      Filter      Wrapper      Embedded
      Methods     Methods       Methods
        │
        ▼
   Correlation
```

Filter methods evaluate features using statistical properties before fitting a predictive model.

Our workflow is:

```text
Feature
   ↓
Correlation with Target
   ↓
Rank
   ↓
Select Top K
```

No machine-learning model needs to be trained to select the features.

---

# 21. Filter vs Wrapper vs Embedded Methods

### Filter Methods

Feature relevance is determined independently of a final predictive model.

Examples:

```text
Correlation
Chi-Square
ANOVA
Mutual Information
```

Advantages:

```text
Fast
Simple
Model-independent
```

### Wrapper Methods

Different feature subsets are evaluated by repeatedly training a model.

Example:

```text
Recursive Feature Elimination (RFE)
```

They can capture model-specific usefulness but are generally more computationally expensive.

### Embedded Methods

Feature selection occurs during model training.

Examples include:

```text
L1/Lasso regularization
Tree-based feature importance
```

---

# 22. Complete Workflow

```text
student_performance.csv
          │
          ▼
     pd.read_csv()
          │
          ▼
Record Missing Values
          │
          ▼
   Identify Data Types
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Numerical  Categorical
     │         │
     ▼         ▼
   Median      Mode
 Imputation  Imputation
     │         │
     └────┬────┘
          │
          ▼
Separate FinalGrade
          │
     ┌────┴────┐
     ▼         ▼
 Features     Target
     │          │
     ▼          │
One-Hot         │
Encoding        │
     │          │
     └────┬─────┘
          ▼
Calculate Correlation
          │
          ▼
Take Absolute Values
          │
          ▼
Sort Descending
          │
          ▼
Select Top 5
          │
          ▼
  Reduced Feature Set
```

---

# 23. Main Functions

The entire project is built around two functions.

### Preprocessing

```python
features, target, missing_before = preprocess_data(
    "student_performance.csv"
)
```

produces clean numerical features.

### Feature Selection

```python
selected_features, correlations = select_features(
    features,
    target,
    top_k=5
)
```

produces the five strongest features according to absolute Pearson correlation.

---

# 24. Key Takeaway

The project performs:

```text
RAW DATA
   ↓
CLEAN DATA
   ↓
NUMERICAL FEATURES
   ↓
MEASURE TARGET RELATIONSHIP
   ↓
RANK FEATURES
   ↓
KEEP TOP K
```

The key distinction is:

> **Feature engineering prepares features; feature selection decides which features to keep.**

In this project, preprocessing makes the data usable, while correlation-based feature selection reduces the feature matrix from **10 features to the top 5 features**.