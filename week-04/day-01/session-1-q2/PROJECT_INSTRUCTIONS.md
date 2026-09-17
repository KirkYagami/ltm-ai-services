# Air Quality Sensor Data — PCA Dimensionality Reduction

## Objective

Build a preprocessing and dimensionality-reduction pipeline for air-quality sensor data.

The project demonstrates:

- Missing-value analysis
- Median imputation
- Feature/target separation
- Feature standardization
- Principal Component Analysis (PCA)
- Explained variance
- Cumulative explained variance
- Automatic component selection
- Dimensionality reduction

---

# Dataset

Input:

```text
air_quality_sensors.csv
```

Target:

```text
AQI_Category
```

The target represents the air-quality classification category and must not be included in PCA.

## Input Features

```text
PM2_5
PM10
NO2
SO2
CO
O3
Temperature
Humidity
WindSpeed
Pressure
```

There are therefore:

```text
10 original features
```

---

# Overall Pipeline

```text
air_quality_sensors.csv
        ↓
Load Data
        ↓
Record Missing Values
        ↓
Median Imputation
        ↓
Separate AQI_Category
        ↓
StandardScaler
        ↓
Standardized Features
        ↓
Full PCA
        ↓
Explained Variance Ratios
        ↓
Cumulative Variance
        ↓
Find Minimum Components ≥ 95%
        ↓
Reduced PCA
        ↓
PC1, PC2, ... PCk
```

---

# Task 1 — `preprocess_data`

## Signature

```python
preprocess_data(filepath)
```

The function must:

1. Load the CSV using Pandas.
2. Record missing-value counts before imputation.
3. Fill missing numerical feature values using their medians.
4. Separate `AQI_Category` from the feature matrix.
5. Standardize all feature columns using `StandardScaler`.
6. Convert the standardized NumPy array back into a DataFrame.
7. Preserve the original feature names.

Return:

```python
scaled_features, target, missing_before
```

---

# Median Imputation

Missing numerical sensor readings are replaced with the median of their respective columns.

```python
data[column] = data[column].fillna(
    data[column].median()
)
```

For example:

```text
PM2_5

10
20
NaN
30
40
```

becomes:

```text
10
20
25
30
40
```

The median is useful because it is less sensitive to extreme sensor readings than the mean.

---

# Why Standardize Before PCA?

The features use very different measurement units.

Examples:

```text
PM2_5       → µg/m³
NO2         → ppb
CO          → ppm
Temperature → °C
Humidity    → %
Pressure    → hPa
```

Without standardization, a feature with a numerically large scale could dominate PCA.

For example:

```text
Pressure ≈ 1000
CO       ≈ 1
```

The difference in numerical scale does not mean pressure is inherently more important.

Therefore use:

```python
StandardScaler()
```

Standardization transforms each feature approximately according to:

```text
z = (x - mean) / standard deviation
```

After scaling, each feature has approximately:

```text
Mean = 0
Standard Deviation = 1
```

This gives features comparable scales before PCA.

---

# Task 2 — `apply_pca`

## Signature

```python
apply_pca(
    scaled_features,
    variance_threshold=0.95
)
```

The function must:

1. Fit PCA using all components.
2. Obtain explained variance ratios.
3. Calculate cumulative explained variance.
4. Find the minimum number of PCs reaching the requested threshold.
5. Fit another PCA using that number of components.
6. Transform the feature matrix.
7. Return the transformed values as a DataFrame.

Return:

```python
reduced_df, pca_full, num_components
```

---

# What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction technique that transforms correlated or redundant features into a new set of mutually orthogonal directions called principal components.

Instead of:

```text
X1
X2
X3
X4
...
X10
```

PCA creates:

```text
PC1
PC2
PC3
...
```

The principal components are ordered according to how much variance they explain.

```text
PC1 → Maximum possible variance

PC2 → Next-highest variance
      perpendicular to PC1

PC3 → Next-highest variance
      perpendicular to PC1 and PC2

...
```

---

# PCA is Feature Extraction

This is different from the previous correlation-based feature-selection exercise.

## Feature Selection

```text
Original:

A B C D E

Select:

A C E
```

Existing columns are retained.

## PCA Feature Extraction

```text
Original:

A B C D E
      ↓
Mathematical transformation
      ↓
PC1 PC2 PC3
```

The PCs are new features constructed from combinations of the original features.

Therefore:

```text
PC1 ≠ one original column
```

Instead, conceptually:

```text
PC1 =
w1 × PM2_5
+ w2 × PM10
+ w3 × NO2
+ ...
+ w10 × Pressure
```

---

# Explained Variance Ratio

PCA tells us how much of the dataset's variance each principal component captures.

Example:

```text
PC1 → 0.40
PC2 → 0.25
PC3 → 0.15
PC4 → 0.10
PC5 → 0.10
```

This means:

```text
PC1 → 40%
PC2 → 25%
PC3 → 15%
...
```

The ratios sum approximately to:

```text
1.0 = 100%
```

---

# Cumulative Explained Variance

We usually care about how much information is retained by several PCs together.

Example:

```text
Component    Variance    Cumulative

PC1          40%         40%
PC2          25%         65%
PC3          15%         80%
PC4          10%         90%
PC5          10%        100%
```

Mathematically:

```text
Cumulative PC3

= PC1 + PC2 + PC3
= 0.40 + 0.25 + 0.15
= 0.80
```

So the first three PCs retain:

```text
80% of total variance
```

---

# Why 95%?

The project uses:

```python
variance_threshold=0.95
```

This asks:

> What is the smallest number of principal components that together preserve at least 95% of the original variance?

We therefore want:

```text
Minimum k such that:

PC1 + PC2 + ... + PCk ≥ 0.95
```

---

# Finding the Number of Components

First:

```python
pca_full = PCA()
pca_full.fit(scaled_features)
```

This calculates all available PCs.

Then:

```python
cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)
```

Suppose:

```text
[0.40, 0.65, 0.80, 0.90, 0.96, 1.00]
```

For:

```text
threshold = 0.95
```

the first value satisfying the requirement is:

```text
0.96
```

which occurs at component 5.

Therefore:

```text
num_components = 5
```

---

# Why `np.searchsorted(...) + 1`?

The implementation uses:

```python
num_components = (
    np.searchsorted(
        cumulative_variance,
        variance_threshold
    )
    + 1
)
```

`searchsorted()` returns a zero-based array position.

Example:

```text
Cumulative:

Index   Value
0       0.40
1       0.65
2       0.80
3       0.90
4       0.96
```

Searching for:

```text
0.95
```

returns:

```text
4
```

But index `4` corresponds to:

```text
5 components
```

Therefore:

```text
4 + 1 = 5
```

---

# Why Fit PCA Twice?

The specification asks us to first determine the explained variance of **all components**.

Therefore:

```python
pca_full = PCA()
pca_full.fit(X)
```

answers:

```text
How much variance does every PC explain?
```

Once the required number is known, we create:

```python
pca_selected = PCA(
    n_components=num_components
)
```

and transform the data.

So:

```text
FULL PCA
   ↓
Analyze Variance
   ↓
Determine k
   ↓
REDUCED PCA(k)
   ↓
Transform Data
```

---

# Expected Dataset Result

The expected output shows:

```text
Original Features: 10
Components for 95% Variance: 9
```

The cumulative variance reaches approximately:

```text
PC8 → 0.8991
PC9 → 0.9959
```

Since:

```text
0.8991 < 0.95
```

eight components are insufficient.

But:

```text
0.9959 ≥ 0.95
```

therefore:

```text
k = 9
```

The transformation becomes:

```text
(10000, 10)
      ↓
(10000, 9)
```

Rows are preserved while columns are reduced.

---

# Principal Component Names

The transformed DataFrame should contain:

```text
PC1
PC2
PC3
...
PC9
```

These are generated using:

```python
[
    f"PC{i}"
    for i in range(1, num_components + 1)
]
```

---

# Important PCA Intuition

Imagine two strongly related features:

```text
PM2_5 ↑
PM10  ↑
```

The points may approximately follow one dominant direction:

```text
PM10
 ^
 |                 *
 |              *
 |           *
 |        *
 |     *
 |  *
 +----------------------> PM2_5
```

Instead of representing this information using two axes, PCA may discover a new axis running along the dominant direction.

```text
Original:

PM2_5 + PM10

       ↓ PCA

      PC1
```

If little information exists in the perpendicular direction, PC2 may explain relatively little variance.

This is how PCA can compress redundant structure.

---

# PCA Does Not Use the Target

Notice:

```python
apply_pca(scaled_features)
```

does not receive:

```text
AQI_Category
```

PCA is **unsupervised**.

It looks at the structure and variance of `X`, not at which features best predict `y`.

This is fundamentally different from the previous correlation-selection exercise:

```text
Correlation Selection:

X + y
 ↓
Find features related to y
```

versus:

```text
PCA:

X only
 ↓
Find directions preserving variance
```

---

# Correlation Selection vs PCA

| Correlation Selection | PCA |
|---|---|
| Feature selection | Feature extraction |
| Keeps original features | Creates new features |
| Uses target | Does not use target |
| Ranks features | Finds variance directions |
| Highly interpretable | PCs can be harder to interpret |
| `StudyHours` remains `StudyHours` | Features become `PC1`, `PC2`, etc. |

---

# Complete Pipeline

```text
RAW SENSOR DATA
       ↓
MEDIAN IMPUTATION
       ↓
SEPARATE TARGET
       ↓
STANDARDIZATION
       ↓
       X
       ↓
FULL PCA
       ↓
EXPLAINED VARIANCE
       ↓
CUMULATIVE VARIANCE
       ↓
Find minimum k ≥ 95%
       ↓
PCA(n_components=k)
       ↓
TRANSFORM
       ↓
PC1 ... PCk
```

---

# Run Project

Install dependencies:

```bash
pip install pandas numpy scikit-learn pytest
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

# Key Takeaways

**Standardization**

```text
Different scales
      ↓
StandardScaler
      ↓
Mean ≈ 0, SD ≈ 1
```

**PCA**

```text
Original Features
      ↓
New orthogonal directions
      ↓
Principal Components
```

**Explained Variance**

```text
How much variance does each PC preserve?
```

**Cumulative Variance**

```text
How much variance do PC1...PCk preserve together?
```

**95% threshold**

```text
Choose the smallest k where:

Cumulative Variance ≥ 0.95
```

The most important conceptual distinction is:

> **Feature selection chooses some of the original features, whereas PCA performs feature extraction and creates entirely new features called principal components.**