"""PCA geometry, reconstruction, and leakage-safe model comparison."""
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def main():
    # An exact, two-dimensional example with sample covariance [[3, 2], [2, 3]].
    mean = np.array([10.0, 20.0])
    directions = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    scores = np.array([[np.sqrt(7.5), 0], [-np.sqrt(7.5), 0],
                       [0, np.sqrt(1.5)], [0, -np.sqrt(1.5)]])
    toy = scores @ directions + mean
    toy_pca = PCA(n_components=1, svd_solver="full").fit(toy)
    toy_rebuilt = toy_pca.inverse_transform(toy_pca.transform(toy))
    toy_error = np.sum((toy - toy_rebuilt) ** 2)
    assert np.allclose(np.cov(toy, rowvar=False), [[3, 2], [2, 3]])
    assert np.isclose(toy_error, 3.0)

    df = pd.read_csv(HERE / "listings.csv")
    X, y = df[["area", "rooms", "rent", "distance"]], df["left_fast"]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    raw = PCA(svd_solver="full").fit(X_dev)
    scaler = StandardScaler().fit(X_dev)
    scaled = scaler.transform(X_dev)
    full = PCA(svd_solver="full").fit(scaled)
    retained = PCA(n_components=0.95, svd_solver="full").fit(scaled)
    weights = pd.DataFrame(full.components_.T, index=X.columns,
                           columns=[f"PC{i+1}" for i in range(4)])
    reconstruction = []
    for k in range(1, 5):
        model = PCA(n_components=k, svd_solver="full").fit(scaled)
        rebuilt = model.inverse_transform(model.transform(scaled))
        sse = np.sum((scaled - rebuilt) ** 2)
        theoretical = (len(scaled) - 1) * full.explained_variance_[k:].sum()
        assert np.isclose(sse, theoretical, atol=1e-8)
        reconstruction.append((k, model.explained_variance_ratio_.sum(),
                               np.mean((scaled - rebuilt) ** 2)))

    candidates = {}
    for name, k in [("no_pca", None), ("pca_1", 1), ("pca_2", 2),
                    ("pca_3", 3), ("pca_4", 4), ("pca_95pct", 0.95)]:
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler())]
        if k is not None:
            steps.append(("pca", PCA(n_components=k, svd_solver="full")))
        steps.append(("model", LogisticRegression(C=1.0, max_iter=2000)))
        candidates[name] = Pipeline(steps)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    splits = list(cv.split(X_dev, y_dev))
    rows = []
    for name, model in candidates.items():
        result = cross_validate(model, X_dev, y_dev, cv=splits,
                                scoring="roc_auc", error_score="raise")
        rows.append((name, result["test_score"].mean(), result["test_score"].std(ddof=1)))
    comparison = pd.DataFrame(rows, columns=["model", "auc_mean", "auc_sd"]).set_index("model")
    # Prefer the first candidate within numerical tolerance of the best mean.
    best = comparison.auc_mean.max()
    winner = next(name for name in candidates if comparison.loc[name, "auc_mean"] >= best - 1e-10)
    final = clone(candidates[winner]).fit(X_dev, y_dev)
    auc = roc_auc_score(y_test, final.predict_proba(X_test)[:, 1])
    report = "\n".join([
        f"Python {platform.python_version()}; NumPy {np.__version__}; pandas {pd.__version__}; scikit-learn {sklearn.__version__}",
        f"Toy sample covariance:\n{np.cov(toy, rowvar=False)}",
        f"Toy first-component variance ratio: {toy_pca.explained_variance_ratio_[0]:.6f}",
        f"Toy reconstruction SSE: {toy_error:.6f}",
        f"\nDevelopment rows: {len(X_dev)}; held-out rows: {len(X_test)}",
        f"Raw PCA variance ratios: {np.array2string(raw.explained_variance_ratio_, precision=6)}",
        f"Standardized PCA variance ratios: {np.array2string(full.explained_variance_ratio_, precision=6)}",
        f"Components retained for 95%: {retained.n_components_}",
        "\nComponent weights (signs may be reversed without changing PCA):",
        weights.to_string(float_format=lambda x: f"{x:.6f}"),
        "\nStandardized reconstruction:",
        pd.DataFrame(reconstruction, columns=["k", "variance_retained", "MSE_per_entry"]).to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "\nRepeated CV ROC AUC:", comparison.to_string(float_format=lambda x: f"{x:.6f}"),
        f"Selected pipeline: {winner}", f"Held-out ROC AUC: {auc:.6f}",
    ])
    (HERE / "pca_results.txt").write_text(report + "\n", encoding="utf-8")
    print(report)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ax = axes[0]
    ax.scatter(toy[:, 0], toy[:, 1], label="Original observations", s=65)
    ax.scatter(toy_rebuilt[:, 0], toy_rebuilt[:, 1], marker="x", s=70,
               color="#c45a24", label="One-component reconstruction")
    for point, rebuilt in zip(toy, toy_rebuilt):
        ax.plot([point[0], rebuilt[0]], [point[1], rebuilt[1]], "--", color="gray")
    t = np.array([-3, 3])
    ax.plot(mean[0] + t / np.sqrt(2), mean[1] + t / np.sqrt(2), color="#23765a", label="PC1 axis")
    ax.set(xlabel="Feature 1", ylabel="Feature 2", title="Projection onto the first principal axis", aspect="equal")
    ax.legend(fontsize=8)
    ax = axes[1]
    indices = np.arange(1, 5)
    ax.bar(indices, full.explained_variance_ratio_, color="#32689a", label="Individual variance ratio")
    ax.plot(indices, full.explained_variance_ratio_.cumsum(), "o-", color="#c45a24", label="Cumulative variance ratio")
    ax.axhline(0.95, linestyle="--", color="gray", label="95% threshold")
    ax.set(xlabel="Principal component", ylabel="Fraction of total variance", title="Standardized rental-listings data", xticks=indices, ylim=(0, 1.06))
    ax.legend(fontsize=8)
    fig.savefig(HERE / "pca_geometry_and_variance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
