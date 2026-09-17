# Product Quality Defect Prediction Using Gradient Boosting

## Objective

Build a machine-learning pipeline that predicts whether newly manufactured products are defective using production sensor measurements.

The project uses:

```text
Gradient Boosting Classifier
```

The complete workflow is:

```text
Historical Training Data
          │
          ▼
     Preprocessing
          │
          ▼
 Gradient Boosting
          │
          ▼
     Trained Model
          │
          ▼
 New Production Data
          │
          ▼
 Defect Predictions
```

The final prediction is:

```text
0 → Not Defective
1 → Defective
```

---

# Files

```text
Project/
│
├── product_quality.csv
├── product_quality_predict.csv
├── main.py
├── tests.py
└── installation.txt
```

The program generates:

```text
predicted_defects.csv
defect_model.joblib
```

---

# Dataset

Training dataset:

```text
product_quality.csv
```

Features:

| Feature | Meaning |
|---|---|
| Temperature | Machine operating temperature |
| Pressure | Production sensor pressure |
| Vibration | Machine vibration intensity |
| RPM | Rotations per minute |
| Humidity | Ambient humidity |
| Voltage | Machine voltage |
| ProcessTime | Processing time |
| MaterialGrade | Raw material grade |
| Defective | Target variable |

Target:

```text
Defective
```

where:

```text
0 → Product is not defective
1 → Product is defective
```

The prediction dataset:

```text
product_quality_predict.csv
```

contains the same input features but does not contain the target column.

---

# Overall Pipeline

```text
product_quality.csv
        │
        ▼
   Load Dataset
        │
        ▼
 Handle Missing Values
        │
   ┌────┴────┐
   ▼         ▼
Numerical  Categorical
 Median       Mode
   └────┬────┘
        ▼
 Separate X and y
        │
        ▼
 One-Hot Encoding
 drop_first=True
        │
        ▼
  StandardScaler
        │
        ▼
 Gradient Boosting
        │
        ▼
    Train Model
        │
        │
product_quality_predict.csv
        │
        ▼
 Apply Same Encoding
        │
        ▼
 Match Training Columns
        │
        ▼
 Use SAME Scaler
        │
        ▼
 model.predict()
        │
        ▼
PredictedDefective
```

---

# Function 1 — `preprocess_data()`

Signature:

```python
preprocess_data(filepath)
```

Its job is to prepare the historical training dataset.

It returns:

```python
(
    scaled_features,
    target,
    scaler,
    feature_columns
)
```

---

# Step 1 — Load Data

```python
data = pd.read_csv(filepath)
```

This produces a Pandas DataFrame.

---

# Step 2 — Handle Missing Numerical Values

The dataset contains missing values in features such as:

```text
Temperature
Vibration
Voltage
```

Numerical missing values are replaced with the median.

Example:

```text
Voltage

215
220
NaN
230
240
```

After median imputation:

```text
215
220
225
230
240
```

Code:

```python
data[column] = data[column].fillna(
    data[column].median()
)
```

Median is generally less sensitive to extreme observations than the mean.

---

# Step 3 — Handle Missing Categorical Values

Categorical missing values are replaced with the mode.

The mode means:

```text
Most frequently occurring value
```

Example:

```text
MaterialGrade

A
B
A
NaN
A
```

Mode:

```text
A
```

Therefore:

```text
NaN → A
```

---

# Step 4 — Separate Features and Target

The target must not be supplied as an input feature.

```python
features = data.drop(
    columns=["Defective"]
)

target = data["Defective"]
```

Conceptually:

```text
             Dataset
                │
        ┌───────┴────────┐
        ▼                ▼
     Features          Target
        X                 y
                          │
                     Defective
```

---

# Step 5 — One-Hot Encoding

`MaterialGrade` is categorical:

```text
A
B
C
```

Machine-learning algorithms require numerical representations.

We use:

```python
pd.get_dummies(
    features,
    drop_first=True
)
```

For example:

```text
MaterialGrade
     │
     ▼
 A   B   C
```

may become:

```text
MaterialGrade_B
MaterialGrade_C
```

because:

```python
drop_first=True
```

removes one reference category.

If:

```text
MaterialGrade_B = 0
MaterialGrade_C = 0
```

the observation belongs to the dropped category `A`.

---

# Why Save `feature_columns`?

After encoding, the exact feature structure is saved:

```python
feature_columns = features.columns.tolist()
```

This is extremely important.

Suppose training produced:

```text
Temperature
Pressure
Vibration
RPM
Humidity
Voltage
ProcessTime
MaterialGrade_B
MaterialGrade_C
```

The model learns this exact input structure.

Prediction data must therefore have:

```text
same columns
same order
same meaning
```

before it is supplied to the model.

---

# Step 6 — StandardScaler

The project requires standardization using:

```python
StandardScaler()
```

Standardization uses:

```text
          x - μ
z = ─────────────
            σ
```

where:

```text
x = original observation
μ = mean
σ = standard deviation
```

For example:

```text
Temperature = 100
Mean        = 80
Std Dev     = 10
```

then:

```text
     100 - 80
z = ──────────
        10

z = 2
```

The observation is therefore two standard deviations above the mean.

---

# Why Return the Scaler?

Training performs:

```python
scaler.fit_transform(features)
```

During `fit`, the scaler learns parameters such as:

```text
mean
standard deviation
```

These same learned values must later be used for prediction.

Therefore the fitted:

```python
scaler
```

is returned from `preprocess_data()`.

---

# Function 2 — `train_boosting()`

Signature:

```python
train_boosting(features, target)
```

The required model is:

```python
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

The model is trained on the complete training dataset:

```python
model.fit(features, target)
```

---

# What Is Gradient Boosting?

Gradient Boosting is an **ensemble learning technique**.

Instead of creating one large model, it creates a sequence of smaller models.

The key idea is:

```text
Model 1 makes predictions
        │
        ▼
Identify mistakes
        │
        ▼
Model 2 tries to correct them
        │
        ▼
Remaining mistakes
        │
        ▼
Model 3 tries to correct them
        │
       ...
        ▼
Combine all models
        │
        ▼
Final Prediction
```

Therefore, unlike bagging where trees are largely constructed independently:

```text
BAGGING

Tree 1 ─┐
Tree 2 ─┼──► Aggregate
Tree 3 ─┘
```

boosting is sequential:

```text
BOOSTING

Tree 1
  ↓
Tree 2
  ↓
Tree 3
  ↓
Tree 4
  ↓
Combined Model
```

Each subsequent learner focuses on improving the errors remaining from the existing ensemble.

---

# `n_estimators=100`

```python
n_estimators=100
```

means Gradient Boosting constructs:

```text
100 boosting stages
```

Conceptually:

```text
Tree 1
  +
Tree 2
  +
Tree 3
  +
...
  +
Tree 100
  =
Final Ensemble
```

Increasing the number of estimators can increase model capacity, but too much capacity can contribute to overfitting.

---

# `max_depth=3`

```python
max_depth=3
```

controls the maximum depth of the individual regression trees used as weak learners.

Conceptually:

```text
                Root
              /      \
           Node      Node
          /   \      /   \
        Node Node  Node  Node
```

A depth of 3 keeps the individual trees relatively small.

Gradient Boosting generally benefits from combining many relatively simple trees rather than relying on one highly complex tree.

---

# `learning_rate=0.1`

The learning rate controls how strongly each boosting stage contributes to the ensemble.

Conceptually:

```text
Final Model
 =
Initial Model
 +
0.1 × Tree 1 contribution
 +
0.1 × Tree 2 contribution
 +
0.1 × Tree 3 contribution
 + ...
```

Smaller learning rates make each individual stage contribute more conservatively.

There is an important relationship between:

```text
learning_rate
      ↕
n_estimators
```

A smaller learning rate commonly requires more boosting stages.

---

# `random_state=42`

```python
random_state=42
```

provides reproducibility for stochastic aspects of the estimator.

The number `42` itself has no special statistical meaning.

---

# How Boosting Differs from Random Forest

```text
RANDOM FOREST

Dataset
 ├── Tree 1
 ├── Tree 2
 ├── Tree 3
 └── Tree N
       │
       ▼
Majority Vote
```

Trees are designed to be diverse and their predictions are aggregated.

Gradient Boosting:

```text
Tree 1
  │
  ▼
Errors
  │
  ▼
Tree 2
  │
  ▼
Remaining Errors
  │
  ▼
Tree 3
  │
  ▼
Combined Prediction
```

A useful conceptual distinction is:

```text
Random Forest
→ Bagging
→ Reduce variance through aggregation

Gradient Boosting
→ Boosting
→ Sequentially improve the ensemble
```

---

# Function 3 — `predict_defects()`

Signature:

```python
predict_defects(
    model,
    filepath,
    scaler,
    feature_columns
)
```

This function applies the trained pipeline to:

```text
product_quality_predict.csv
```

---

# Prediction Pipeline

```text
New Production Data
        │
        ▼
 Handle Missing Data
        │
        ▼
 One-Hot Encoding
        │
        ▼
 Match Training Columns
        │
        ▼
 Training Scaler
        │
        ▼
 Gradient Boosting
        │
        ▼
 PredictedDefective
```

---

# Critical Concept — Training/Prediction Alignment

Suppose training produced:

```text
Temperature
Pressure
Vibration
RPM
Humidity
Voltage
ProcessTime
MaterialGrade_B
MaterialGrade_C
```

But a new prediction batch happens to contain only:

```text
MaterialGrade A
MaterialGrade B
```

After `get_dummies()`, it might not naturally contain:

```text
MaterialGrade_C
```

This would create a mismatch.

The solution is:

```python
data = data.reindex(
    columns=feature_columns,
    fill_value=0
)
```

This guarantees:

```text
TRAINING COLUMNS
        =
PREDICTION COLUMNS
```

in both names and order.

---

# Why `fill_value=0`?

Suppose the model expects:

```text
MaterialGrade_C
```

but the current prediction batch contains no Grade C products.

The encoded column can therefore be:

```text
MaterialGrade_C

0
0
0
0
0
```

The column still exists so that the model receives the expected feature structure.

---

# Critical Concept — Never Fit a New Scaler

Training:

```python
scaler.fit_transform(training_features)
```

Prediction:

```python
scaler.transform(prediction_features)
```

NOT:

```python
scaler.fit_transform(prediction_features)
```

Why?

Because the model was trained in the coordinate system created by the training scaler.

Correct:

```text
TRAINING DATA
      │
      ▼
Scaler.fit()
      │
      ▼
Learn μ and σ
      │
      ├─────────────┐
      ▼             ▼
Training Data   Prediction Data
      │             │
 transform()     transform()
      │             │
      └──────┬──────┘
             ▼
       Same Scale
```

Incorrect:

```text
Training Data       Prediction Data
     │                    │
Scaler A              Scaler B
     │                    │
     ▼                    ▼
Different coordinate systems
```

---

# Generating Predictions

After preprocessing:

```python
predictions = model.predict(
    scaled_data
)
```

The result contains:

```text
0
1
0
0
1
...
```

where:

```text
0 → Not Defective
1 → Defective
```

---

# Preserve Original Prediction Data

Before transforming the prediction dataset:

```python
original_data = data.copy()
```

The model operates on the transformed version.

However, the output should contain the original readable values.

Therefore:

```python
original_data["PredictedDefective"] = predictions
```

produces something conceptually like:

```text
Temperature | Pressure | MaterialGrade | PredictedDefective
-------------------------------------------------------------
   81.5     |  105.2   |       A       |         0
   96.7     |  120.4   |       C       |         1
   84.1     |  101.8   |       B       |         0
```

---

# Saving Predictions

The final DataFrame is written using:

```python
result.to_csv(
    "predicted_defects.csv",
    index=False
)
```

Result:

```text
predicted_defects.csv
```

The expected output contains all original prediction columns plus:

```text
PredictedDefective
```

---

# Saving the Model

The trained estimator is saved using:

```python
joblib.dump(
    model,
    "defect_model.joblib"
)
```

This creates:

```text
defect_model.joblib
```

A saved model can later be loaded using:

```python
model = joblib.load(
    "defect_model.joblib"
)
```

without retraining the Gradient Boosting model.

---

# Important Practical Note

Tree-based models such as Gradient Boosting generally do not require `StandardScaler` in the way that distance- or gradient-sensitive algorithms often do.

Decision-tree splits are based on thresholds:

```text
Temperature < 85?
```

Changing units or linearly scaling the feature generally does not fundamentally alter the ordering used by such splits.

However, this exercise explicitly requires:

```text
StandardScaler
```

so it must be included in this implementation.

---

# Important Production Pipeline Consideration

This exercise returns:

```text
model
scaler
feature_columns
```

separately.

Conceptually:

```text
RAW DATA
   │
   ▼
Encoding
   │
   ▼
Scaler
   │
   ▼
Model
```

All preprocessing artifacts are part of the effective prediction pipeline.

In a production system, these transformations are commonly packaged together using tools such as:

```python
Pipeline
```

and:

```python
ColumnTransformer
```

This reduces the risk of training-serving preprocessing mismatches.

---

# Generated Files

After running:

```bash
python3 main.py
```

the project should contain:

```text
Project/
│
├── product_quality.csv
├── product_quality_predict.csv
├── main.py
├── tests.py
├── installation.txt
│
├── predicted_defects.csv
└── defect_model.joblib
```

---

# Run Tests

```bash
python3 -m pytest tests.py -v
```

The tests verify that:

```text
preprocess_data()
    ✓ callable
    ✓ no missing values
    ✓ no object features

train_boosting()
    ✓ callable
    ✓ returns model

predict_defects()
    ✓ callable
    ✓ returns PredictedDefective
    ✓ preserves prediction row count

Outputs
    ✓ predicted_defects.csv can be created
    ✓ defect_model.joblib can be created
```

---

# Key Concepts to Remember

```text
Median Imputation
→ Fill numerical missing values

Mode Imputation
→ Fill categorical missing values

One-Hot Encoding
→ Convert categorical features to numeric columns

StandardScaler
→ Standardize feature values

Gradient Boosting
→ Sequential ensemble learning

n_estimators
→ Number of boosting stages

max_depth
→ Maximum individual tree depth

learning_rate
→ Contribution of each boosting stage

feature_columns
→ Preserve exact training feature structure

reindex()
→ Align prediction features with training features

scaler.transform()
→ Apply training scaling to new observations

model.predict()
→ Generate defect classifications

joblib
→ Persist trained model
```

The most important end-to-end idea is:

```text
                    TRAINING
                       │
Raw Historical Data ───┤
                       ▼
              Learn Preprocessing
                       │
                       ▼
                 Learn Model
                       │
                       ▼
                Trained System
                       │
                       │
                    PREDICTION
                       │
New Raw Data ──────────┤
                       ▼
          Apply SAME Preprocessing
                       │
                       ▼
               Apply SAME Model
                       │
                       ▼
              Defect Prediction
```

**Training learns the transformations and the model; prediction only applies what was learned.**