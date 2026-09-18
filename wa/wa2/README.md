# ORDER RETURN RISK PREDICTION

## PROBLEM STATEMENT

A national e-commerce retailer ships orders from four regional fulfilment centres. Its operations desk can hold, repack or insure an order while it is still on the packing bench, so a return-risk warning issued before dispatch carries operational value.

The retailer's order register is fed by the fulfilment centres and arrives with duplicated records, unreported readings and inconsistently spelled shipping methods. An automated pipeline is required to turn this register into a return-risk indicator and apply it to the orders awaiting dispatch.

## OBJECTIVE

Build a three-stage pipeline that prepares the raw order register for modelling, trains a classifier to identify orders at risk of being returned, and applies the stored classifier to an unlabelled batch of pending orders.

## FILE STRUCTURE

```
├── data/
│   ├── order_return_train.csv
│   └── order_return_predict.csv
├── processed_data/
├── artifacts/
├── output/
├── main.py
└── requirements.txt
```

## PURPOSE OF FILES

| File | Purpose |
|------|---------|
| `data/order_return_train.csv` | Raw register with recorded return outcomes. |
| `data/order_return_predict.csv` | Raw batch awaiting dispatch, with no outcome. |
| `processed_data` | Destination for the model-ready registers. |
| `artifacts` | Destination for the trained classifier. |
| `output` | Destination for the issued return-risk indicators. |
| `main.py` | Module to be implemented. |
| `requirements.txt` | Pinned package versions. |

## DATASET DESCRIPTION

### Training Dataset:

**SCHEMA**

| Column | Description |
|--------|-------------|
| order_id | Unique order identifier |
| order_date | Date the order was placed |
| center_code | Fulfilment centre code |
| destination_city | Destination city |
| carrier_partner | Carrier partner name |
| package_weight_kg | Package weight in kilograms |
| item_price | Price of the item |
| shipping_method | Shipping method (standard, express, priority) |
| is_gift_order | Whether the order is a gift (no, yes) |
| is_returned | Outcome: whether the order was returned |

**SAMPLE RECORDS**

| order_id | order_date | center_code | destination_city | carrier_partner | package_weight_kg | item_price | shipping_method | is_gift_order | is_returned |
|----------|------------|-------------|------------------|-----------------|-------------------|------------|-----------------|---------------|-------------|
| ORD001 | 2024-01-15 | C01 | Mumbai | BlueDart | 2.5 | 1500.0 | standard | no | 0 |
| ORD002 | 2024-01-16 | C02 | Delhi | Delhivery | 1.2 | 800.0 | express | yes | 1 |
| ORD003 | 2024-01-17 | C01 | Bangalore | BlueDart | 3.8 | 2500.0 | priority | no | 0 |

## TASK

### MODULE EXPLANATION

Implement `main.py` as three independent functions and a main block that runs them in sequence. Every location a function works with arrives as an argument, and each function must operate from those arguments alone.

The cleaning stage is shared by the training register and the pending batch, which keeps the columns presented to the classifier aligned with those it was fitted on. It must not depend on the return outcome being present. No function prints; all reporting belongs to the main block.

### DATA CLEANING

**FUNCTION SIGNATURE**

```python
def clean_data(input_path: str, output_path: str) -> pd.DataFrame:
    ...
```

**PARAMETERS**

- `input_path` is a string locating a raw order register in `data`.
- `output_path` is a string locating the model-ready register to be written in `processed_data`.

**REQUIREMENTS**

- A model-ready table holds only predictive columns, no duplicate records, no gaps and numeric values throughout.
- The five booking columns `order_id`, `order_date`, `center_code`, `destination_city` and `carrier_partner` are not predictive.
- Duplicate records must be removed before any summary statistic is drawn, since each unreported reading in `package_weight_kg` and `item_price` is repaired with its own column's median.
- Text in `shipping_method` and `is_gift_order` must reach a single lowercase form without surrounding spaces before conversion, where `standard`, `express` and `priority` become `0`, `1` and `2`, and `no` and `yes` become `0` and `1`.

**RETURNS**

- The model-ready register as a pandas DataFrame, matching the file written to `output_path`.

### MODEL TRAINING

**FUNCTION SIGNATURE**

```python
def train_model(processed_path: str, model_path: str) -> RandomForestClassifier:
    ...
```

**PARAMETERS**

- `processed_path` is a string locating the model-ready training register in `processed_data`.
- `model_path` is a string locating the fitted classifier to be stored in `artifacts`.

**REQUIREMENTS**

- `is_returned` is the outcome and must be held apart from the inputs.
- The estimator is a random forest fixed at 100 trees, a maximum depth of 10 and a random state of 42, so that every submission yields an identical model.
- It is fitted on the whole model-ready register and persisted to `model_path`.

**RETURNS**

- The fitted random forest classifier.

### PREDICTION

**FUNCTION SIGNATURE**

```python
def predict(model_path: str, input_path: str, processed_path: str, output_path: str) -> pd.DataFrame:
    ...
```

**PARAMETERS**

- `model_path` is a string locating the stored classifier in `artifacts`.
- `input_path` is a string locating the raw pending batch in `data`.
- `processed_path` is a string locating the model-ready form of that batch in `processed_data`.
- `output_path` is a string locating the indicators to be written in `output`.

**REQUIREMENTS**

- The stored classifier is loaded, not refitted.
- The pending batch is prepared by the same cleaning stage used for training.
- Cleaning removes `order_id`, so the references must be held from the raw batch, and each must remain paired with the indicator issued for its own record.
- Every order receives exactly one indicator.
- The result carries `order_id` and `predicted_is_returned` only.

**RETURNS**

- A pandas DataFrame holding `order_id` and `predicted_is_returned`.

### MAIN BLOCK

The main block runs the three stages in sequence and is the only place that prints. It reports:

1. The dimensions of the model-ready training register.
2. The number of inputs the classifier was fitted on.
3. The dimensions of the issued indicators.

## OUTPUT

### GENERATED FILES

| File | Description |
|------|-------------|
| `processed_data/order_return_train_cleaned.csv` | Model-ready training register. |
| `processed_data/order_return_predict_cleaned.csv` | Model-ready pending batch. |
| `artifacts/return_risk_model.pkl` | Fitted random forest classifier. |
| `output/order_return_predictions.csv` | Return-risk indicator for each pending order. |

### CONSOLE OUTPUT

```
Training register dimensions: (rows, columns)
Number of inputs fitted: N
Issued indicators dimensions: (rows, columns)
```

## IMPLEMENTATION STEPS

1. **STEP 1** — Move into the project directory.
2. **STEP 2** — Install the pinned packages.
3. **STEP 3** — Run the pipeline.
4. **STEP 4** — To run the testcases, click the **Run Testcases** button or in CLI run the command.