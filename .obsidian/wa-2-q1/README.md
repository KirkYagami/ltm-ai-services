# Shipment Delivery Delay Prediction — Setup & Run Guide

## Overview

This project implements a three-stage machine learning pipeline for a national logistics operator:

1. **Data Cleaning** — Turn raw shipment registers into model-ready tables.
2. **Model Training** — Fit a random forest classifier on the cleaned training data.
3. **Prediction** — Apply the stored classifier to an unlabelled batch of pending shipments.

The pipeline is orchestrated by `main.py` and produces four artifacts: two cleaned CSV files, one trained model, and one prediction file.

---

## Project Structure

```
Project/
├── data/
│   ├── shipment_delivery_train.csv      # Raw register with delivery outcomes
│   └── shipment_delivery_predict.csv    # Raw batch awaiting dispatch (no outcome)
├── processed_data/                       # Destination for model-ready registers
├── artifacts/                            # Destination for the trained classifier
├── output/                               # Destination for issued delay indicators
├── main.py                               # Pipeline implementation
├── requirements.txt                      # Pinned package versions
└── tests.py                              # Test suite
```

> If any of the output folders (`processed_data/`, `artifacts/`, `output/`) do not exist, they will be created automatically by the pipeline.

---

## Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (Python package installer)
- A terminal / shell (bash, zsh, PowerShell, or equivalent)

All Python dependencies are pinned in `requirements.txt`. The core libraries used are:

- `pandas` — tabular data handling
- `numpy` — numerical operations
- `scikit-learn` — random forest classifier and validation utilities
- `joblib` — model persistence
- `pytest` — test runner

---

## Setup Instructions

### Step 1 — Move into the project directory

```bash
cd Project
```

### Step 2 — (Optional but recommended) Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install the pinned packages

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 4 — Execute the full pipeline

```bash
python3 main.py
```

This runs the three stages in sequence:

1. **Cleaning stage** — reads `data/shipment_delivery_train.csv`, writes the model-ready form to `processed_data/shipment_delivery_train_cleaned.csv`.
2. **Training stage** — fits a `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)` on the cleaned training register and persists it to `artifacts/delivery_delay_model.pkl`.
3. **Prediction stage** — cleans `data/shipment_delivery_predict.csv`, loads the stored model, and writes `order_id` + `predicted_is_delayed` to `output/shipment_delivery_predictions.csv`.

### Expected Console Output

The `main` block is the only place that prints. After a successful run you should see three lines similar to:

```
Model-ready training register dimensions: (N, 8)
Number of inputs the classifier was fitted on: 7
Issued indicator dimensions: (M, 2)
```

Where:

- `N` is the number of unique, cleaned training records (7 features + `is_delayed`).
- `7` is the number of feature columns the classifier was fitted on.
- `M` is the number of pending consignments scored (must equal the raw prediction batch size).

---

## Generated Files

| Path | Description |
| --- | --- |
| `processed_data/shipment_delivery_train_cleaned.csv` | Model-ready training register |
| `processed_data/shipment_delivery_predict_cleaned.csv` | Model-ready pending batch |
| `artifacts/delivery_delay_model.pkl` | Fitted random forest classifier |
| `output/shipment_delivery_predictions.csv` | Delay indicator for each pending consignment |

---

## Running the Tests

### Step 5 — Run the test suite

From the CLI:

```bash
python3 -m pytest tests.py -v
```

Alternatively, use the **Run Testcases** button in the IDE.

All tests must pass before submission. The test suite validates:

- `clean_data` returns a DataFrame and writes a matching file.
- Metadata / identifier columns are removed; feature columns and `is_delayed` (if present) are kept.
- Duplicate records are removed **before** medians are computed.
- Missing `distance_km` and `package_weight_kg` values are filled with the **post-dedup column median**.
- `shipping_speed` and `is_weekend_dispatch` are normalised (lowercase, trimmed) and encoded correctly.
- The prediction path works when `is_delayed` is absent.
- `train_model` writes a loadable `RandomForestClassifier` with the exact required hyperparameters.
- `train_model` excludes `is_delayed` from the training features.
- `train_model` learns the pattern in the data (beats a majority baseline).
- `predict_new_data` loads the model from the given path (not a cached one).
- Every row in the pending batch receives exactly one indicator, paired with its own `order_id`.
- Output contains exactly `order_id` and `predicted_is_delayed`, with labels in `{0, 1}` (not probabilities).

---

## Implementation Rules (for reference)

These constraints are enforced by the test suite and must be respected when editing `main.py`:

### `clean_data(input_path, output_path)`

- Returns the cleaned `pandas.DataFrame` and writes it to `output_path`.
- Drops the five booking columns: `order_id`, `dispatch_date`, `origin_warehouse_code`, `destination_city`, `courier_partner`.
- Removes exact duplicate rows **before** computing any median.
- Fills missing `distance_km` and `package_weight_kg` with their own column's **median**.
- Normalises `shipping_speed` and `is_weekend_dispatch` to lowercase, stripped form before encoding:
  - `standard → 0`, `express → 1`, `priority → 2`
  - `no → 0`, `yes → 1`
- Must **not** reference `is_delayed` — the prediction batch has no such column.
- Output is entirely numeric.

### `train_model(processed_path, model_path)`

- Holds `is_delayed` apart from the inputs.
- Uses `RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)`.
- Fits on the **whole** model-ready register.
- Persists via `joblib.dump` to `model_path`.
- Returns the fitted classifier.

### `predict_new_data(model_path, input_path, processed_path, output_path)`

- Loads the stored classifier (never refits).
- Cleans the pending batch with the **same** `clean_data` function used for training.
- Reads `order_id` from the **raw** batch (since cleaning removes it) and pairs each id with its own indicator.
- Output holds only `order_id` and `predicted_is_delayed`.
- Uses `model.predict(...)` (labels), **not** `model.predict_proba(...)`.

### `main()`

- Runs the three stages in sequence.
- Is the **only** place in the module that prints.
- Reports:
  1. Dimensions of the model-ready training register.
  2. Number of inputs the classifier was fitted on.
  3. Dimensions of the issued indicators.

---

## Troubleshooting

| Symptom | Likely Cause / Fix |
| --- | --- |
| `FileNotFoundError: data/shipment_delivery_train.csv` | Run the command from the project root (`Project/`), not from a subfolder. |
| `ModuleNotFoundError: No module named 'sklearn'` | Dependencies were not installed. Run `pip install -r requirements.txt`. |
| Test fails with "median was taken while repeated records were still present" | Call `drop_duplicates()` **before** the `fillna(median)` step. |
| Test fails with "clean_data must not create is_delayed" | Do not add or reference `is_delayed` inside `clean_data`. |
| Test fails with "model was trained with is_delayed as an input" | Drop `is_delayed` from `X` before calling `model.fit(X, y)`. |
| Test fails with "probabilities or text labels are not accepted" | Use `model.predict(...)`, not `predict_proba(...)`. |
| Predictions are constant (all 0s or all 1s) | The model was likely not loaded from `model_path`, or features were misaligned. Verify column order matches `FEATURE_COLUMNS`. |
| `KeyError: 'order_id'` during prediction | Capture `order_id` from the **raw** batch before cleaning, since cleaning removes it. |

---

## Submission Checklist

- [ ] `main.py` is implemented with the three required functions.
- [ ] `pip install -r requirements.txt` completed without errors.
- [ ] `python3 main.py` runs end-to-end and prints the three required lines.
- [ ] All four output files are generated in their respective directories.
- [ ] `python3 -m pytest tests.py -v` passes **all** tests.
- [ ] The **Submit Project** button has been clicked at least once (required for submission).

---

## Notes

- The classifier is fully deterministic (`random_state=42`), so every submission produces an identical model given identical training data.
- Do **not** commit generated files (`processed_data/`, `artifacts/`, `output/`) unless explicitly required — they are reproduced by running `python3 main.py`.
- If you modify `clean_data`, verify that the same logic works for both the training register (which has `is_delayed`) and the prediction batch (which does not).