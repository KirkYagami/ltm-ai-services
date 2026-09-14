# Ride Fare Prediction - Random Forest Regression

## Objective

Build a Random Forest Regression pipeline that predicts ride fares
using pickup location, drop-off location, trip distance, trip duration,
and day of the week.

The solution must:

- Clean missing values.
- Encode categorical variables.
- Train a Random Forest Regression model.
- Use bootstrap aggregation (bagging).
- Generate predictions for new rides.
- Save the trained model and prediction results.

---

## Project Structure

```text
Project/
├── ride_data.csv
├── new_rides.csv
├── main.py
├── tests.py
├── installation.txt
├── cleaned_ride_data.csv
├── random_forest_model.joblib
└── predicted_fares.csv