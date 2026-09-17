# Student Outcome Prediction
## Understanding Early Stopping with SGDClassifier

This project demonstrates **early stopping** using Scikit-learn's `SGDClassifier`.

We train two otherwise similar classifiers:

```text
Model A → No Early Stopping
Model B → Early Stopping Enabled
```

and compare:

```text
Iterations
Training Accuracy
Testing Accuracy
Train-Test Gap
```

---

# 1. What is SGDClassifier?

`SGDClassifier` is a linear classifier trained using **Stochastic Gradient Descent (SGD)**.

Instead of calculating an update using the entire training dataset at once, SGD updates the model incrementally using individual training observations.

Conceptually:

```text
Initialize model weights
        ↓
Take training observation
        ↓
Make prediction
        ↓
Calculate loss
        ↓
Calculate gradient
        ↓
Update weights
        ↓
Next observation
        ↓
Repeat...
```

The basic update can be represented as:

```text
New Weight = Old Weight - Learning Rate × Gradient
```

or mathematically:

```math
w_{new} = w_{old} - η∇L(w)
```

where:

```text
w  = model weights
η  = learning rate
L  = loss function
∇L = gradient of the loss
```

---

# 2. Why loss="log_loss"?

The project creates:

```python
SGDClassifier(loss="log_loss")
```

`log_loss` makes SGDClassifier optimize the logistic regression objective.

For binary classification:

```math
P(y=1|x) = 1 / (1 + e^(-z))
```

where:

```math
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Therefore, this project can be viewed as **logistic regression trained using stochastic gradient descent**.

---

# 3. What is an Iteration?

In `SGDClassifier`, `max_iter` represents the maximum number of passes over the training data.

One complete pass through the training dataset is commonly called an **epoch**.

Therefore:

```python
max_iter=5000
```

means the model is allowed to make up to:

```text
5000 complete passes
through the training dataset.
```

The actual number of epochs completed after fitting is available through:

```python
model.n_iter_
```

---

# 4. Why Scale the Features?

SGD is sensitive to feature scale.

Consider:

```text
StudyHoursPerWeek → 1 - 40
AttendanceRate    → 30 - 100
PreviousGPA       → 1 - 4
```

These features operate at very different numerical magnitudes.

The project therefore uses:

```python
StandardScaler()
```

which approximately calculates:

```math
z = (x - μ) / σ
```

where:

```text
x = original value
μ = feature mean
σ = feature standard deviation
```

After scaling:

```text
StudyHours ─────┐
Attendance ─────┼──→ Comparable numerical scales
GPA ────────────┘
```

This generally makes gradient-based optimization more stable.

---

# 5. Missing Values

Approximately 3% of the feature values are missing.

Numerical features use:

```python
SimpleImputer(strategy="mean")
```

Example:

```text
GPA

3.1
2.8
NaN
3.4
```

The missing value is replaced by the mean.

Categorical features use:

```python
SimpleImputer(strategy="most_frequent")
```

Therefore, a missing category is replaced with the most common category.

---

# 6. One-Hot Encoding

Features such as:

```text
CourseLevel
LearningMode
```

contain strings.

For example:

```text
CourseLevel

Beginner
Intermediate
Advanced
```

A machine-learning model needs numerical representations.

`OneHotEncoder` converts categories into binary features.

For example:

```text
              Beginner   Intermediate
Advanced          0            0
Beginner          1            0
Intermediate      0            1
```

The project uses:

```python
OneHotEncoder(drop="first")
```

so one category is used as the reference category.

---

# 7. What is Early Stopping?

Early stopping means:

> Stop training when additional training is no longer producing sufficient improvement on validation data.

Without early stopping:

```text
Training
   ↓
Epoch 1
   ↓
Epoch 2
   ↓
Epoch 3
   ↓
...
   ↓
Epoch 5000
```

The algorithm continues even if additional training is no longer useful.

With early stopping:

```text
Training
   ↓
Performance improves
   ↓
Performance improves
   ↓
Performance improves
   ↓
Improvement plateaus
   ↓
Wait for allowed non-improving epochs
   ↓
STOP
```

The model may therefore stop long before reaching `max_iter`.

---

# 8. Validation Data

When:

```python
early_stopping=True
```

is enabled, `SGDClassifier` reserves part of the training data for validation.

The project specifies:

```python
validation_fraction=0.1
```

Therefore:

```text
X_train
   │
   ├────────────── 90% → Weight updates
   │
   └────────────── 10% → Validation
```

The validation observations are not used for updating the model coefficients during fitting.

Instead, they are used to determine whether training should continue.

This is important because checking only training performance could encourage the model to keep fitting the training observations.

---

# 9. n_iter_no_change

The project uses:

```python
n_iter_no_change=10
```

Conceptually this means:

```text
Validation performance
        ↓
No sufficient improvement
        ↓
1 epoch
2 epochs
3 epochs
...
10 epochs
        ↓
STOP
```

So the algorithm does not stop immediately after one disappointing epoch.

It allows several epochs without sufficient improvement before terminating.

---

# 10. max_iter

Both models have:

```python
max_iter=5000
```

But `max_iter` means:

> Maximum permitted number of training epochs.

It does NOT necessarily mean:

> The model must always execute 5000 epochs.

For the early-stopping model:

```text
Maximum = 5000

Actual training:

1 → 2 → 3 → ... → 16 → STOP
```

Therefore:

```python
model.n_iter_
```

might report:

```text
16
```

even though:

```python
max_iter == 5000
```

---

# 11. Why Does the Non-Early-Stopping Model Run 5000 Iterations?

The first model specifies:

```python
tol=None
```

This disables convergence-based stopping.

Therefore:

```text
max_iter = 5000
tol = None

        ↓

No convergence stopping

        ↓

Run all 5000 epochs
```

This deliberately creates the contrast required by the exercise.

---

# 12. alpha = 1e-10

The models use:

```python
alpha=1e-10
```

`alpha` controls the strength of the regularization term in `SGDClassifier`.

```text
Large alpha
    ↓
Stronger regularization

Small alpha
    ↓
Weaker regularization
```

Here:

```text
alpha = 0.0000000001
```

which is extremely small.

Therefore, regularization is intentionally negligible for this demonstration.

This makes the experiment focus primarily on the effect of **early stopping**.

---

# 13. Training Without Early Stopping

Conceptually:

```text
Loss
 ^
 |\
 | \
 |  \
 |   \________
 |            \_ fluctuations
 |
 +--------------------------> Epochs
                            5000
```

The optimizer continues updating the coefficients throughout all allowed epochs.

Continuing optimization does not guarantee improved unseen-data performance.

---

# 14. Training With Early Stopping

Conceptually, validation performance may behave like:

```text
Validation
Performance
 ^
 |             *
 |          *     *
 |       *
 |    *
 | *
 +----------------------------> Epoch
              ↑
          useful region

                  ─────→ no meaningful
                         improvement
                              ↓
                             STOP
```

The model stops once validation performance has failed to improve sufficiently for the configured number of epochs.

---

# 15. Train-Test Accuracy Gap

The project compares:

```text
Training Accuracy
        -
Testing Accuracy
        =
Accuracy Gap
```

For example:

```text
Train Accuracy = 0.7210
Test Accuracy  = 0.7097

Gap = 0.7210 - 0.7097
    = 0.0113
```

A larger positive gap can indicate that the model performs better on training observations than unseen observations.

However, the gap should never be interpreted alone.

---

# 16. Why Can the Gap Be Negative?

The example output shows:

```text
Train Accuracy = 0.7764
Test Accuracy  = 0.7773
```

Therefore:

```text
0.7764 - 0.7773
= -0.0009
```

This is completely possible.

It simply means that the measured test accuracy happened to be slightly higher than the measured training accuracy.

It does **not** mean the calculation is wrong.

The difference is extremely small:

```text
0.09 percentage points
```

so train and test performance are essentially very close.

---

# 17. Example Result

The exercise shows approximately:

```text
WITHOUT EARLY STOPPING

Iterations       5000
Train Accuracy   72.10%
Test Accuracy    70.97%
Gap               1.13%
```

while:

```text
WITH EARLY STOPPING

Iterations         16
Train Accuracy   77.64%
Test Accuracy    77.73%
Gap              -0.09%
```

The striking difference is:

```text
5000 epochs
     vs
16 epochs
```

while the example early-stopping run also obtains higher test accuracy.

For this particular generated dataset and configuration, early stopping therefore demonstrates both:

```text
Training Efficiency
        +
Better observed test performance
```

These results should not be interpreted as a universal guarantee that early stopping always produces higher test accuracy.

---

# 18. Early Stopping and Overfitting

The general idea is:

```text
                    MODEL TRAINING

Undertrained       Useful Training        Excessive Training
     │                    │                       │
     ▼                    ▼                       ▼

High Bias          Better Fit             Overfitting Risk
     │                    │                       │
     └────────────────────┼───────────────────────┘
                          ▲
                     Desired region
```

Early stopping attempts to terminate optimization around the point where validation performance stops improving.

It therefore acts as a form of **regularization**.

---

# 19. Bias-Variance Connection

As training continues, a model may increasingly adapt to the training data.

Conceptually:

```text
Too little training
        ↓
Underfitting
        ↓
High Bias


Appropriate training
        ↓
Useful balance
        ↓
Good Generalization


Excessive fitting
        ↓
Greater sensitivity to training data
        ↓
Potential High Variance
```

Early stopping attempts to avoid unnecessary movement toward the last region.

---

# 20. Complete Project Workflow

```text
student_outcome_data.csv
          │
          ▼
       Pandas
          │
          ▼
    Separate X and y
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 Numerical  Categorical
     │         │
     ▼         ▼
   Mean       Mode
 Imputation Imputation
     │         │
     ▼         ▼
 Standard   One-Hot
  Scaling   Encoding
     │         │
     └────┬────┘
          │
          ▼
   ColumnTransformer
          │
          ▼
   Processed Dataset
          │
          ▼
    80/20 Train-Test
          │
      ┌───┴───────────┐
      │               │
      ▼               ▼
SGDClassifier     SGDClassifier
No Early Stop     Early Stopping
      │               │
 5000 epochs       Validation
      │            monitored
      │               │
      │          Stop when
      │          improvement
      │           plateaus
      │               │
      └───────┬───────┘
              ▼
          Evaluation
              │
     ┌────────┼─────────┐
     ▼        ▼         ▼
   Train     Test    Iterations
 Accuracy  Accuracy
     │        │
     └───┬────┘
         ▼
  Train-Test Gap
```

---

# 21. Important Hyperparameters

| Parameter | Purpose |
|---|---|
| `loss="log_loss"` | Logistic regression loss |
| `alpha=1e-10` | Very weak regularization |
| `max_iter=5000` | Maximum training epochs |
| `tol=None` | Forces non-early-stop model to run all epochs |
| `early_stopping=True` | Enables validation-based stopping |
| `validation_fraction=0.1` | Uses 10% of training data for validation |
| `n_iter_no_change=10` | Stops after insufficient improvement for multiple epochs |
| `random_state=42` | Makes randomized behavior reproducible |

---

# 22. Main Takeaway

Without early stopping:

```text
Training continues
      ↓
5000 epochs
      ↓
More computation
      ↓
Potential unnecessary optimization
```

With early stopping:

```text
Monitor validation performance
             ↓
Detect lack of improvement
             ↓
Stop training
             ↓
Fewer epochs
             +
Potentially better generalization
```

The central lesson is:

> **More training is not automatically better training.**

Early stopping uses validation performance to determine when continued optimization is no longer beneficial.