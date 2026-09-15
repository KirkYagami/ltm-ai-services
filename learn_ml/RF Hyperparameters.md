### 🌲 Random Forest — Hyperparameter Master Table

| Hyperparameter                                                                                     | Classifier Default | Regressor Default | Explanation / What exactly it controls                                                                                                                                                                                  | Practical tuning intuition                                                                                                                                                                                 |
| -------------------------------------------------------------------------------------------------- | -----------------: | ----------------: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`n_estimators`** ⭐                                                                               |              `100` |             `100` | **Number of decision trees in the forest.** Each tree is independently trained using its own sampling/randomness. Final classification combines class probabilities across trees; regression averages tree predictions. | ↑ More trees → generally **more stable predictions and lower variance**, but more computation/memory. More trees usually don't cause classic overfitting in the same way deeper trees can.                 |
| **`max_features`** ⭐⭐⭐                                                                             |           `"sqrt"` |             `1.0` | **Number of randomly selected features considered at each node/split.** A new random subset is selected **again at every node of every tree**. The best split is then found only among those candidate features.        | ↓ Fewer features → trees become **more diverse / less correlated**, but individual trees may become weaker. ↑ More features → potentially stronger individual trees but more correlation between trees.    |
| **`bootstrap`** ⭐⭐⭐                                                                                |             `True` |            `True` | Determines whether each tree receives a **bootstrap sample** of the training rows. `True` = rows are randomly sampled **with replacement**, so rows can repeat and others can be absent from a particular tree.         | Usually keep `True`. This creates **row-level randomness** and diversity between trees. Setting `False` makes every tree use the entire dataset, though feature randomness can still make trees different. |
| **`max_samples`** ⭐⭐<br><br>- control bootstrap sample size<br>- 0.8 -> 80%<br>- 5000 -> 5000 rows |             `None` |            `None` | Controls **how many rows are drawn for each tree's bootstrap sample** when `bootstrap=True`. `None` means the number of draws equals the original training-set size.                                                    | Example: `0.8` → each tree gets bootstrap draws equal to **80% of N**. Smaller values increase diversity and reduce training cost, but each tree learns from less data.                                    |
| **`max_depth`** ⭐⭐⭐                                                                                |             `None` |            `None` | Maximum allowed **depth of each individual decision tree**. `None` means there is no explicit depth limit; trees grow until other stopping conditions stop them.                                                        | ↓ Smaller depth → simpler trees, ↓ variance, ↑ bias. Very small values can **underfit**. Deep trees are common in Random Forest because averaging helps control their variance.                            |
| **`min_samples_split`** ⭐⭐                                                                         |                `2` |               `2` | Minimum number of samples a node must contain before it is **eligible to split**. With `2`, even a node containing only two samples may potentially be split.                                                           | ↑ Increase it to prevent trees from creating splits from very small groups of observations. Higher values → **simpler trees** and can reduce overfitting.                                                  |
| **`min_samples_leaf`** ⭐⭐⭐                                                                         |                `1` |               `1` | Minimum number of samples that must remain in **each resulting leaf** after a split. A proposed split is rejected if either child would contain fewer than this number.                                                 | One of the most useful regularization controls. ↑ Larger values → smoother, less-specific predictions and **less complex trees**.                                                                          |
| **`criterion`** ⭐⭐                                                                                 |           `"gini"` | `"squared_error"` | Defines **how the algorithm evaluates the quality of candidate splits**. Classifier default uses Gini impurity; regressor default uses reduction in squared error.                                                      | Usually leave at default unless experimentation/CV shows another criterion performs better. It controls **how "best split" is mathematically determined**.                                                 |
| **`max_leaf_nodes`**                                                                               |             `None` |            `None` | Maximum number of **terminal/leaf nodes** allowed in each tree. `None` means no explicit maximum.                                                                                                                       | ↓ Restricting leaves limits tree complexity. Example: `max_leaf_nodes=20` forces each tree to have at most 20 leaves.                                                                                      |
| **`min_impurity_decrease`**                                                                        |              `0.0` |             `0.0` | A node will split only when the proposed split produces at least this much **weighted impurity reduction**.                                                                                                             | ↑ Higher threshold → reject weak splits → **smaller/simpler trees**. Usually left at `0.0` unless deliberately regularizing trees.                                                                         |
| **`oob_score`** ⭐⭐                                                                                 |            `False` |           `False` | Enables **Out-of-Bag evaluation**. Rows not included in a particular tree's bootstrap sample can evaluate that tree. Predictions are combined across trees where an observation was OOB. Requires `bootstrap=True`.     | `True` gives a useful internal estimate of generalization performance without requiring those OOB observations to train the corresponding trees. Useful for Random Forest diagnostics.                     |
| **`random_state`** ⭐⭐                                                                              |             `None` |            `None` | Controls the random-number generation used by Random Forest, including randomness involved in bootstrap sampling and feature selection.                                                                                 | Set an integer such as `42` when you need **reproducible results**. Same data + parameters + seed → reproducible forest.                                                                                   |
| **`n_jobs`** ⭐                                                                                     |             `None` |            `None` | Controls how many CPU jobs can run in parallel for operations such as fitting/predicting. Trees are well suited to parallel processing.                                                                                 | `n_jobs=-1` → use **all available processors**. Mainly affects **speed**, not model quality.                                                                                                               |
| **`warm_start`**                                                                                   |            `False` |           `False` | If `True`, calling `fit()` again can **reuse the existing fitted forest and add additional trees** rather than starting completely from scratch, when configured appropriately.                                         | Useful when incrementally increasing `n_estimators`, e.g. 100 → 200 → 300 trees. Usually `False`.                                                                                                          |
| **`ccp_alpha`**                                                                                    |              `0.0` |             `0.0` | Complexity parameter for **Minimal Cost-Complexity Pruning** of individual trees. `0.0` means effectively no pruning through this mechanism.                                                                            | ↑ Larger values → more aggressive pruning → **smaller trees**. Less commonly tuned for Random Forest than `max_depth`/`min_samples_leaf`.                                                                  |
| **`class_weight`**                                                                                 |             `None` |                 — | **Classifier only.** Assigns different importance/weights to classes during training. Useful when one class is much rarer than another.                                                                                 | For imbalanced classification, options such as `"balanced"` or `"balanced_subsample"` can increase the importance of minority-class observations.                                                          |
| **`monotonic_cst`**                                                                                |             `None` |            `None` | Allows **monotonic constraints** on features, where predictions can be constrained to increase or decrease as a particular feature increases.                                                                           | Advanced feature. Useful when domain knowledge requires a predictable directional relationship between a feature and prediction.                                                                           |

[scikit-learn RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?utm_source=chatgpt.com)  
[scikit-learn RandomForestRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?utm_source=chatgpt.com)

### 🧠 The parameters I'd memorize first

The table is comprehensive, but conceptually you can organize the important parameters into four groups:

```text
🌲 RANDOM FOREST
│
├── 🌳 FOREST
│     └── n_estimators
│         "How many trees?"
│
├── 🎲 RANDOMNESS
│     ├── bootstrap
│     │   "Sample rows with replacement?"
│     │
│     ├── max_samples
│     │   "How many row draws per tree?"
│     │
│     └── max_features ⭐
│         "How many random features per NODE?"
│
├── ✂️ TREE COMPLEXITY
│     ├── max_depth
│     ├── min_samples_split
│     ├── min_samples_leaf ⭐
│     ├── max_leaf_nodes
│     └── min_impurity_decrease
│
└── ⚙️ OTHER
      ├── criterion       → How is "best split" measured?
      ├── oob_score       → Use OOB observations for evaluation?
      ├── random_state    → Reproducibility?
      ├── n_jobs          → Parallel CPU usage?
      ├── warm_start      → Add trees later?
      └── ccp_alpha       → Prune trees?
```

The **four parameters that explain the architecture of Random Forest particularly well** are:

> **`n_estimators`** → How many trees do I build?  
> **`bootstrap` / `max_samples`** → What random rows does each tree receive?  
> **`max_features`** → What random features can each node consider?  
> **Tree-complexity parameters** → How large/deep can each individual tree become?

And one distinction worth memorizing:

> **`max_samples` = randomness across ROWS at the TREE level**  
> **`max_features` = randomness across FEATURES at the NODE level**

That distinction is particularly useful in interviews because it shows you understand _where_ Random Forest actually injects randomness.