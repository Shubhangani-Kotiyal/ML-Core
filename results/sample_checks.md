# Sample comparison checks

Run on the six included datasets after reorganising the repository. The C++ build completed with -std=c++11 -Wall -Wextra -pedantic. All six options completed in both integrated menus. Python reference tested with scikit-learn 1.8.0.

| Algorithm | Inputs | C++ | Python reference |
|---|---|---|---|
| Regression | x=7 | 13.2667; training MSE 0.184127 | 13.2666667; training MSE 0.184126984 |
| K-Means | K=2 | Centers (1.75,1.625), (5.75,5.625) | Same centers |
| SVM | (2,2) | Class 1; 8/8 correct training predictions | Class 1; training accuracy 1.0 |
| Hierarchical | 2 groups | Points 1-3 and 4-6 | Same grouping |
| PCA | 2D to 1D | Explained variance ratio 0.963181 | 0.963181314 |
| KNN | K=3, (2,2) | Class 0 | Class 0 |

This is a small sample comparison, not proof of correctness for all datasets. No held-out test accuracy is claimed. SVM weights differ because the optimisation and regularisation settings differ. Cluster numbers and PCA signs can differ without changing the solution. Equidistant KNN neighbours can appear in different orders.

Reproduce by running make, then ./ml_core, or python3 python/ml_core_python.py, and selecting each option with the inputs above.
