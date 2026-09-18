"""Run from any directory: python path/to/listings_demo.py."""
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
RAW = ["area", "rooms", "rent", "distance"]
FEATURES = RAW + ["rent_per_area", "area_per_room"]


def make_listings():
    # Educational simulation: these are invented relationships, not market data.
    rng = np.random.default_rng(42)
    n = 800
    area = rng.uniform(350, 1800, n).round(1)
    rooms = np.clip(np.rint(area / 450 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    distance = rng.uniform(0.5, 25, n).round(2)
    unit_rent = rng.uniform(18, 55, n)
    rent = (area * unit_rent).round(0)
    log_odds = 0.35 - 0.13 * (rent / area - 35) - 0.10 * (distance - 10)
    probability = 1 / (1 + np.exp(-log_odds))
    left_fast = rng.binomial(1, probability)
    return pd.DataFrame(dict(area=area, rooms=rooms, rent=rent,
                             distance=distance, left_fast=left_fast))


def add_features(X):
    out = X.copy()
    # Invalid denominators become missing; never silently divide by zero.
    out["rent_per_area"] = out["rent"] / out["area"].where(out["area"] > 0)
    out["area_per_room"] = out["area"] / out["rooms"].where(out["rooms"] > 0)
    return out


def logistic():
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)


def make_pipeline(engineered=False, selector=None):
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_features, validate=False)))
    steps.extend([("impute", SimpleImputer(strategy="median")),
                  ("scale", StandardScaler())])
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("model", logistic()))
    return Pipeline(steps)


def main():
    csv_path = HERE / "listings.csv"
    if not csv_path.exists():
        make_listings().to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    X, y = df[RAW], df["left_fast"]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)

    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "raw_scaled": make_pipeline(),
        "engineered_scaled": make_pipeline(engineered=True),
        "engineered_kbest3": make_pipeline(
            engineered=True, selector=SelectKBest(f_classif, k=3)),
        "engineered_rfe3": make_pipeline(
            engineered=True, selector=RFE(logistic(), n_features_to_select=3, step=1)),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    splits = list(cv.split(X_dev, y_dev))  # Identical folds for each candidate.
    scoring = {"auc": "roc_auc", "accuracy": "accuracy",
               "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, pipeline in candidates.items():
        result = cross_validate(pipeline, X_dev, y_dev, cv=splits,
                                scoring=scoring, error_score="raise", n_jobs=1)
        rows.append({"model": name,
                     "auc_mean": result["test_auc"].mean(),
                     "auc_sd": result["test_auc"].std(ddof=1),
                     "accuracy": result["test_accuracy"].mean(),
                     "balanced_accuracy": result["test_balanced_accuracy"].mean()})
    results = pd.DataFrame(rows).set_index("model")
    winner = results["auc_mean"].idxmax()
    final_model = clone(candidates[winner]).fit(X_dev, y_dev)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = final_model.predict(X_test)

    # Inspection uses development data only, never the final test labels.
    engineered = add_features(X_dev)
    pearson = engineered.assign(left_fast=y_dev).corr()["left_fast"].drop("left_fast")
    spearman = engineered.assign(left_fast=y_dev).corr(method="spearman")["left_fast"].drop("left_fast")
    filter_model = clone(candidates["engineered_kbest3"]).fit(X_dev, y_dev)
    kbest = filter_model.named_steps["select"]
    filter_table = pd.DataFrame({"feature": FEATURES, "pearson_r": pearson.reindex(FEATURES).values,
                                 "spearman_rho": spearman.reindex(FEATURES).values,
                                 "F": kbest.scores_, "p_value": kbest.pvalues_,
                                 "kept": kbest.get_support()})
    rfe_model = clone(candidates["engineered_rfe3"]).fit(X_dev, y_dev)
    rfe = rfe_model.named_steps["select"]
    rfe_table = pd.DataFrame({"feature": FEATURES, "rank": rfe.ranking_, "kept": rfe.support_})
    coefs = pd.Series(rfe_model.named_steps["model"].coef_[0],
                      index=np.array(FEATURES)[rfe.support_])
    new_listing = pd.DataFrame([dict(area=900, rooms=2, rent=27000, distance=4)])
    report = "\n".join([
        f"Python {platform.python_version()}; NumPy {np.__version__}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}",
        f"Rows: {len(df)}; development: {len(X_dev)}; test: {len(X_test)}",
        f"Development class counts: {y_dev.value_counts().sort_index().to_dict()}",
        "\nRepeated CV (15 validation scores per candidate):", results.to_string(float_format=lambda x: f"{x:.4f}"),
        f"\nSelected by development CV AUC: {winner}",
        f"Held-out AUC: {roc_auc_score(y_test, probabilities):.4f}",
        f"Held-out accuracy: {accuracy_score(y_test, predictions):.4f}",
        f"Held-out balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}",
        "\nDevelopment-data filter inspection:", filter_table.to_string(index=False, float_format=lambda x: f"{x:.6g}"),
        "\nDevelopment-data RFE inspection:", rfe_table.to_string(index=False),
        "\nFinal logistic coefficients in the RFE pipeline (standardized features):", coefs.to_string(),
        f"\nNew-listing P(left_fast=1): {final_model.predict_proba(new_listing)[0, 1]:.4f}",
    ])
    print(report)
    (HERE / "results.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
