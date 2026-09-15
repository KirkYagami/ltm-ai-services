import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# 1. Generate a nonlinear classification dataset
# ============================================================

X, y = make_moons(
    n_samples=200,
    noise=0.20,
    random_state=42
)

# Scale features - very important for SVM
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ============================================================
# 2. Gamma values to compare
# ============================================================

gamma_values = [
    0.001,
    0.01,
    0.1,
    1,
    10
]

# Keep C constant so that we isolate the effect of gamma
C = 1.0


# ============================================================
# 3. Create grid for plotting decision regions
# ============================================================

x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1

y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1


xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]


# ============================================================
# 4. Create one figure containing multiple subplots
# ============================================================

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(24, 14)
)

axes = axes.flatten()


# ============================================================
# 5. Train one RBF SVM for each gamma
# ============================================================

for ax, gamma in zip(axes, gamma_values):

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma
    )

    model.fit(X, y)


    # --------------------------------------------------------
    # Predict decision function across entire feature space
    # --------------------------------------------------------

    Z = model.decision_function(grid)

    Z = Z.reshape(xx.shape)


    # --------------------------------------------------------
    # Draw predicted regions
    # --------------------------------------------------------

    ax.contourf(
        xx,
        yy,
        Z,
        levels=30,
        alpha=0.25,
        cmap="coolwarm"
    )


    # --------------------------------------------------------
    # Draw decision boundary
    #
    # f(x) = 0
    # --------------------------------------------------------

    ax.contour(
        xx,
        yy,
        Z,
        levels=[0],
        linewidths=2.5,
        colors="black"
    )


    # --------------------------------------------------------
    # Plot training observations
    # --------------------------------------------------------

    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette="coolwarm",
        s=80,
        edgecolor="black",
        ax=ax,
        legend=False
    )


    # --------------------------------------------------------
    # Highlight actual support vectors
    # --------------------------------------------------------

    support_vectors = model.support_vectors_

    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=220,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Support Vectors"
    )


    # --------------------------------------------------------
    # Model statistics
    # --------------------------------------------------------

    training_accuracy = model.score(X, y)

    number_of_support_vectors = len(
        model.support_
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    ax.set_title(
        f"Gamma = {gamma}\n"
        f"Accuracy = {training_accuracy:.3f} | "
        f"Support Vectors = {number_of_support_vectors}",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Scaled Feature 1",
        fontsize=12
    )

    ax.set_ylabel(
        "Scaled Feature 2",
        fontsize=12
    )

    ax.legend(
        loc="upper right"
    )


# ============================================================
# 6. Use final subplot to explain gamma mathematically
# ============================================================

ax = axes[-1]


# Distance values
distance = np.linspace(
    0,
    4,
    500
)


for gamma in gamma_values:

    # RBF Kernel:
    #
    # K(x, x') = exp(-gamma * ||x - x'||²)

    similarity = np.exp(
        -gamma * distance**2
    )

    ax.plot(
        distance,
        similarity,
        linewidth=2.5,
        label=f"γ = {gamma}"
    )


ax.set_title(
    "RBF Influence vs Distance",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel(
    "Distance between observations",
    fontsize=12
)

ax.set_ylabel(
    "RBF Similarity",
    fontsize=12
)

ax.legend()

ax.grid(
    alpha=0.3
)


# ============================================================
# 7. Main figure title
# ============================================================

fig.suptitle(
    "Effect of Gamma (γ) on RBF Support Vector Machine",
    fontsize=26,
    fontweight="bold",
    y=1.02
)


fig.text(
    0.5,
    0.975,
    "Same Dataset | Same C = 1.0 | Only Gamma Changes",
    ha="center",
    fontsize=16
)


# ============================================================
# 8. Layout
# ============================================================

plt.tight_layout(
    rect=[0, 0, 1, 0.95]
)


# ============================================================
# 9. Save as high-resolution PNG
# ============================================================

plt.savefig(
    "svm_gamma_comparison.png",
    dpi=500,
    bbox_inches="tight"
)


# ============================================================
# 10. Display
# ============================================================

plt.show()