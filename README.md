# ML-Core

An educational C++ project implementing Linear Regression, K-Means, SVM, Hierarchical Clustering, PCA and KNN from scratch using classes, fixed-size arrays and shared mathematical functions.

## Build and run

From the repository root with a C++11 compiler and make installed:

```bash
make
./ml_core
```

Without make:

```bash
g++ -std=c++11 -Wall -Wextra -pedantic -Iinclude src/*.cpp -o ml_core
./ml_core
```

Compile all seven source files together. Only src/ml_algorithms_menu.cpp defines main(). All six classes inherit from MLAlgorithm. The menu calls execute() through base-class pointers, demonstrating runtime polymorphism. include/MLAlgorithm.h owns the common arrays and CSV loader; include/MathUtils.h provides shared calculations.

## Datasets

CSV files live in datasets/. Run the C++ program from the repository root. Use commas, no header, finite numerical values and at most 100 rows.

| Algorithm | Source | Dataset | Columns |
|---|---|---|---|
| Linear Regression | src/linear_regression.cpp | datasets/linear_regression.csv | x,y |
| K-Means | src/kmeans.cpp | datasets/kmeans.csv | x,y |
| Linear SVM | src/svm.cpp | datasets/svm.csv | x,y,label (-1 or 1) |
| Hierarchical Clustering | src/hierarchical_clustering.cpp | datasets/hierarchical_clustering.csv | x,y |
| PCA | src/pca.cpp | datasets/pca.csv | x,y |
| KNN | src/knn.cpp | datasets/knn.csv | x,y,integer label |

Regression uses direct least squares for one input feature. SVM is linear and binary. Clustering and KNN use two features. Hierarchical clustering uses single linkage. PCA centers two features and projects onto one component.

## Sample checks

The integrated menu has been compiled and exercised with all six included datasets.

| Option | Input | Expected result |
|---|---|---|
| Regression | x=7 | Prediction about 13.2667; training MSE 0.184127 |
| K-Means | K=2 | Centroids (1.75,1.625) and (5.75,5.625) |
| SVM | Point 2 2 | Class 1; 8/8 correct training predictions |
| Hierarchical | 2 clusters | First three points and last three points form separate groups |
| PCA | No further input | Explained variance ratio about 0.963181 |
| KNN | K=3, point 2 2 | Class 0 |

These are small demonstration checks, not comprehensive validation. Training accuracy does not measure performance on unseen data. A small comparison against scikit-learn is recorded in results/sample_checks.md. Broader evaluation and charts remain pending. Invalid keyboard extraction ends the program; restart to try again.

## Team workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch purposes and commands. Every working branch contains the complete project.

## Reports and presentations

- reports/phase1/: report source and PDF.
- reports/phase2/: report PDF.
- reports/phase3/: placeholder; Phase 3 report still needs uploading.
- presentations/phase1/: Phase 1 source and PDF; later presentations still need uploading.

## Team

- Shubhangani Kotiyal
- Ishanaya Agarwal
- Harshita Joshi

Each member should commit their own work with their own Git identity.

## Project folders

| Folder | Contents |
|---|---|
| src/ | C++ classes and integrated menu |
| include/ | Shared mathematical functions |
| datasets/ | Six numerical datasets |
| python/ | Scikit-learn reference menu and dependencies |
| results/ | Recorded sample checks |
| reports/ | Phase-wise reports |
| presentations/ | Phase-wise presentations |

## Python reference program

```bash
python3 -m pip install -r python/requirements.txt
python3 python/ml_core_python.py
```

This reference uses NumPy and scikit-learn; the C++ implementation remains from scratch. Python resolves datasets relative to the script location. Cluster labels and PCA signs can differ without changing the solution. SVM uses a different solver and regularization setting, so its parameters need not match the C++ model.

See [sample results](results/sample_checks.md) for smoke-test results.

## Inheritance and CSV input update

After choosing an algorithm, enter - to use its sample CSV or enter a custom CSV path. Paths can contain spaces. C++ paths are relative to the current working directory; run from the repository root.

The shared C++ loader reads one physical line at a time and accepts at most 100 nonblank data rows, with no header. Exactly 100 rows are accepted; larger files are rejected rather than truncated. Missing fields, extra columns, nonfinite numbers and invalid labels are rejected. Blank lines are ignored. Regression and PCA require at least two rows; SVM requires both -1 and 1 classes. Other algorithms require at least one row. K-Means still supports at most 10 clusters; its dataset may contain 100 rows. Python also enforces the 100-row maximum.

The inheritance/loader refactor has been source-reviewed but has not yet been compiled or executed in this editing session because the execution environment was unavailable. Earlier sample results refer to the preceding implementation. Run make and recheck all menu options after pulling.

Boundary-check script: run make test to exercise all six C++ algorithms with 99, 100, and 101 rows, malformed rows, and paths containing spaces. This new script is supplied but has not been executed in this session.
