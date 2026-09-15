
## C

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------------------------------------------------
# 1. Generate a simple 2D classification dataset
# ---------------------------------------------------------

X, y = make_blobs(
    n_samples=80,
    centers=[(-1.4, -1.0), (1.4, 1.0)],
    cluster_std=1.25,
    random_state=12
)


# ---------------------------------------------------------
# 2. Scale the features
# ---------------------------------------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ---------------------------------------------------------
# 3. Different values of C to compare
# ---------------------------------------------------------

C_values = [0.01, 1, 10, 1000]


# ---------------------------------------------------------
# 4. Train and visualize one SVM for each C
# ---------------------------------------------------------

for C in C_values:

    # Train a linear SVM
    model = SVC(
        kernel="linear",
        C=C
    )

    model.fit(X, y)


    # -----------------------------------------------------
    # Create a dense grid covering the feature space
    # -----------------------------------------------------

    x1 = np.linspace(
        X[:, 0].min() - 1,
        X[:, 0].max() + 1,
        350
    )

    x2 = np.linspace(
        X[:, 1].min() - 1,
        X[:, 1].max() + 1,
        350
    )

    xx, yy = np.meshgrid(x1, x2)

    grid = np.c_[
        xx.ravel(),
        yy.ravel()
    ]


    # -----------------------------------------------------
    # Calculate SVM decision function for every grid point
    # -----------------------------------------------------

    Z = model.decision_function(grid)

    Z = Z.reshape(xx.shape)


    # -----------------------------------------------------
    # Create plot
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))


    # Class 0
    ax.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        s=65,
        marker="o",
        label="Class 0"
    )


    # Class 1
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        s=65,
        marker="s",
        label="Class 1"
    )


    # -----------------------------------------------------
    # Draw decision boundary and margins
    #
    # f(x) = -1  → margin
    # f(x) =  0  → decision boundary
    # f(x) = +1  → margin
    # -----------------------------------------------------

    ax.contour(
        xx,
        yy,
        Z,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
        linewidths=[1.5, 2.8, 1.5]
    )


    # -----------------------------------------------------
    # Highlight support vectors
    # -----------------------------------------------------

    support_vectors = model.support_vectors_

    ax.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=190,
        facecolors="none",
        linewidths=2,
        label="Support vectors"
    )


    # -----------------------------------------------------
    # Plot information
    # -----------------------------------------------------

    training_accuracy = model.score(X, y)

    ax.set_title(
        f"Linear SVM: C={C} | "
        f"Train Accuracy={training_accuracy:.3f} | "
        f"Support Vectors={len(support_vectors)}"
    )

    ax.set_xlabel("Scaled Feature 1")
    ax.set_ylabel("Scaled Feature 2")

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    plt.show()
```