# E-Commerce Transaction Outlier Detection and PCA Variance Comparison

## Objective

Build a preprocessing pipeline that:

- Handles missing transaction data
- Detects outliers using the IQR method
- Caps extreme values instead of deleting observations
- Standardizes features
- Applies Principal Component Analysis (PCA)
- Compares PCA explained variance before and after outlier treatment

---

# Dataset

```text
ecommerce_transactions.csv
```

## Features

```text
SessionDuration
PagesViewed
CartValue
DiscountUsed
PreviousPurchases
DaysSinceLastVisit
ItemsInCart
ShippingFee
```

Target:

```text
HighSpender
```

The target is excluded from outlier processing and PCA.

Therefore:

```text
9 total columns
       ↓
Remove HighSpender
       ↓
8 feature columns
```

---

# Complete Pipeline

```text
ecommerce_transactions.csv
            ↓
       Load Dataset
            ↓
 Record Missing Counts
            ↓
    Median Imputation
            ↓
    Separate HighSpender
            ↓
      Clean Features
        /         \
       /           \
      ↓             ↓
Raw Features    IQR Detection
                    ↓
                Cap Outliers
                    ↓
             Capped Features
      ↓             ↓
StandardScaler  StandardScaler
      ↓             ↓
     PCA           PCA
      ↓             ↓
Variance Before  Variance After
       \            /
        \          /
         ↓        ↓
       Compare Top 3 PCs
```

---

# Function 1 — `preprocess_data`

## Signature

```python
preprocess_data(filepath)
```

The function performs three main operations:

```text
Load
 ↓
Impute
 ↓
Separate X and y
```

Return:

```python
features, target, missing_before
```

where:

- `features` contains the eight predictor columns
- `target` contains `HighSpender`
- `missing_before` records missing-value counts before imputation

---

# Missing-Value Treatment

The dataset contains missing values in columns such as:

```text
SessionDuration
PagesViewed
CartValue
```

Missing numerical values are replaced using the column median.

```python
data[column] = data[column].fillna(
    data[column].median()
)
```

Example:

```text
CartValue

10
20
30
NaN
500
```

Sorted:

```text
10 20 30 500
```

Median:

```text
(20 + 30) / 2 = 25
```

Therefore:

```text
NaN → 25
```

Median imputation is particularly useful when data may contain extreme values because the median is less sensitive to outliers than the mean.

---

# Function 2 — `detect_outliers`

## Signature

```python
detect_outliers(features)
```

The function uses the **Interquartile Range (IQR)** method.

For every numerical feature:

```text
1. Calculate Q1
2. Calculate Q3
3. Calculate IQR
4. Calculate lower bound
5. Calculate upper bound
6. Count observations outside bounds
7. Cap them to the boundaries
```

Return:

```python
capped_features, outlier_counts
```

---

# Understanding Quartiles

Consider:

```text
10 12 14 15 16 18 20 22 25 100
```

The value:

```text
100
```

looks unusually large compared with the rest.

Quartiles divide the ordered observations into sections.

```text
             middle 50%
        <---------------->
----|----------|----------|---------
   Q1        Median       Q3
```

`Q1` is approximately the 25th percentile.

`Q3` is approximately the 75th percentile.

---

# Interquartile Range

The IQR measures the width of the middle 50% of the observations.

```text
IQR = Q3 - Q1
```

Suppose:

```text
Q1 = 20
Q3 = 40
```

Then:

```text
IQR = 40 - 20
    = 20
```

---

# IQR Outlier Boundaries

The conventional 1.5 × IQR rule defines:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Using:

```text
Q1  = 20
Q3  = 40
IQR = 20
```

we get:

```text
Lower = 20 - (1.5 × 20)
      = -10

Upper = 40 + (1.5 × 20)
      = 70
```

Therefore:

```text
x < -10  → lower outlier
x > 70   → upper outlier
```

A value of:

```text
100
```

is therefore classified as an outlier.

---

# Outlier Detection in Python

```python
outlier_mask = (
    (features[column] < lower_bound)
    |
    (features[column] > upper_bound)
)
```

Then:

```python
outlier_mask.sum()
```

counts the number of outlying observations.

The results are stored as:

```python
outlier_counts[column] = count
```

Conceptually:

```python
{
    "SessionDuration": 543,
    "PagesViewed": 134,
    "CartValue": 565,
    ...
}
```

---

# Outlier Capping

This project does **not delete outliers**.

Instead, it caps them.

Suppose:

```text
Lower Bound = 10
Upper Bound = 100
```

Original:

```text
5
25
50
80
130
```

After capping:

```text
10
25
50
80
100
```

Therefore:

```text
5   → 10
130 → 100
```

while normal observations remain unchanged.

---

# Why Use `clip()`?

Pandas provides:

```python
Series.clip()
```

Therefore:

```python
features[column].clip(
    lower=lower_bound,
    upper=upper_bound
)
```

means:

```text
value < lower bound
        ↓
replace with lower bound

value inside bounds
        ↓
leave unchanged

value > upper bound
        ↓
replace with upper bound
```

---

# Why Cap Instead of Delete?

Deleting an outlier removes an entire observation.

```text
Outlier row
    ↓
DELETE
    ↓
Information lost
```

Capping instead performs:

```text
Extreme value
    ↓
Bring to boundary
    ↓
Row retained
```

This can be useful when an observation is legitimate but its magnitude would disproportionately influence subsequent analysis.

---

# Important: IQR Is Calculated Per Feature

Every feature receives its own boundaries.

For example:

```text
CartValue
Q1 = ...
Q3 = ...
IQR = ...
Bounds = ...

SessionDuration
Q1 = ...
Q3 = ...
IQR = ...
Bounds = ...

DaysSinceLastVisit
Q1 = ...
Q3 = ...
IQR = ...
Bounds = ...
```

There is no single global IQR boundary for the entire DataFrame.

---

# Function 3 — `compare_pca`

## Signature

```python
compare_pca(
    raw_features,
    capped_features,
    top_n=3
)
```

This function determines whether outlier treatment changes how PCA distributes variance among the principal components.

Return:

```python
variance_before, variance_after
```

Both are NumPy arrays.

For:

```python
top_n=3
```

they contain:

```text
PC1
PC2
PC3
```

explained-variance ratios.

---

# Why Standardize Before PCA?

The e-commerce variables have different units.

```text
SessionDuration     → minutes
PagesViewed         → count
CartValue           → USD
DiscountUsed        → percentage
PreviousPurchases   → count
DaysSinceLastVisit  → days
ItemsInCart         → count
ShippingFee         → USD
```

Suppose:

```text
CartValue = 500

ItemsInCart = 4
```

The larger numerical magnitude of `CartValue` should not automatically make it dominate PCA.

Therefore we standardize:

```python
StandardScaler()
```

using:

```text
z = (x - μ) / σ
```

This produces features with approximately:

```text
mean = 0
standard deviation = 1
```

---

# Why Two Separate StandardScalers?

This detail is important.

We have:

```text
Raw Features
```

and:

```text
Capped Features
```

Capping changes the distributions.

Therefore each dataset gets its own fitted scaler:

```python
raw_scaler = StandardScaler()

capped_scaler = StandardScaler()
```

Then:

```python
raw_scaler.fit_transform(raw_features)
```

and:

```python
capped_scaler.fit_transform(capped_features)
```

This compares the correlation/variance structure of each version after putting its features on comparable standardized scales.

---

# PCA Comparison

We fit:

```python
pca_before = PCA()
pca_after = PCA()
```

Then:

```text
RAW
 ↓
Standardize
 ↓
PCA
 ↓
Explained Variance Before
```

and:

```text
CAPPED
 ↓
Standardize
 ↓
PCA
 ↓
Explained Variance After
```

---

# Explained Variance Ratio

Suppose PCA produces:

```text
PC1 = 0.30
PC2 = 0.20
PC3 = 0.15
```

This means:

```text
PC1 → 30% of variance
PC2 → 20% of variance
PC3 → 15% of variance
```

Together:

```text
0.30 + 0.20 + 0.15
= 0.65
= 65%
```

The project compares these ratios before and after capping.

---

# Why Can Outliers Affect PCA?

PCA searches for directions with maximum variance.

Consider mostly compact observations:

```text
      * *
    * * *
     * *
```

and one extreme observation:

```text
      * *
    * * *
     * *


                         *
                       outlier
```

The extreme observation increases variance strongly in its direction.

PCA may therefore rotate a principal component toward that observation.

Conceptually:

```text
Extreme observations
        ↓
Change covariance/correlation structure
        ↓
Change PCA directions
        ↓
Change explained variance ratios
```

After capping:

```text
Extreme observations
        ↓
IQR boundaries
        ↓
Reduced extreme influence
        ↓
Recalculate PCA
```

This is exactly what the exercise investigates.

---

# Why PCA Is Fitted With All Components

The requirement says:

```text
fit a full PCA
```

Therefore:

```python
PCA()
```

is used instead of:

```python
PCA(n_components=3)
```

We then extract:

```python
pca.explained_variance_ratio_[:3]
```

The analysis therefore computes the full decomposition but reports only the requested first three components.

---

# Expected Comparison

The supplied example shows approximately:

```text
Component    Before      After

PC1          0.1288      0.1314
PC2          0.1279      0.1270
PC3          0.1275      0.1262
```

Notice that the explained variance ratios change after capping.

That demonstrates:

```text
Outlier Treatment
       ↓
Changes feature distributions
       ↓
Changes multivariate structure
       ↓
Can change PCA results
```

---

# Outlier Count vs Number of Outlier Rows

Be careful with:

```text
Total outliers = 1959
```

This is the sum of per-column outlier counts.

It does **not necessarily mean 1959 unique rows** contain outliers.

For example:

```text
Row 10:
SessionDuration → outlier
CartValue       → outlier
```

That single row contributes:

```text
2
```

to the summed feature-level outlier count.

---

# Why `HighSpender` Must Be Excluded

`HighSpender` is the target:

```text
0 → Not high spender
1 → High spender
```

It should not participate in:

```text
IQR capping
StandardScaler
PCA
```

Therefore:

```python
target = data["HighSpender"]

features = data.drop(
    columns=["HighSpender"]
)
```

This also prevents target leakage into the feature transformation.

---

# Key Difference From the Previous PCA Exercise

The previous exercise asked:

```text
How many PCs are required
to preserve ≥95% variance?
```

This exercise asks:

```text
How does outlier treatment
change PCA explained variance?
```

So the goals differ:

```text
Previous Exercise
PCA → Dimensionality Reduction

This Exercise
Outliers → Capping → PCA Comparison
```

---

# Run

Install dependencies:

```bash
pip install pandas scikit-learn pytest
```

Run the project:

```bash
python3 main.py
```

Run tests:

```bash
python3 -m pytest tests.py -v
```

---

# Key Takeaways

```text
MEDIAN
→ handles missing numerical values robustly

IQR
→ detects unusually low/high observations

CAPPING
→ limits extreme values without deleting rows

STANDARDIZATION
→ prevents feature units/scales from dominating PCA

PCA
→ finds directions of maximum variance

EXPLAINED VARIANCE RATIO
→ measures the fraction of variance captured by each PC
```

The central idea of this exercise is:

> **PCA is driven by variance. Extreme observations can influence that variance, so comparing PCA before and after IQR capping helps reveal the effect of outlier treatment on the structure of the data.**