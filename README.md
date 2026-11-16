# ML-Core

A student project implementing machine learning algorithms from scratch in C++ using fixed-size arrays and standard library headers.

## Current repository status

- LinearRegression.cpp: standalone class-based simple linear regression, CSV input, direct least-squares fitting, prediction and training MSE.
- MathUtils.h: shared mean, line fitting, prediction, MSE and two-dimensional K-Means distance/centroid helpers.
- ml_algorithms_menu.cpp: earlier menu-driven implementation. It has separate implementations; it is not yet connected to the new classes. Its PCA option only performs centering and covariance calculation.
- Separate class-based K-Means, KNN, SVM, Hierarchical Clustering and PCA files are to be added by the team.
- Comparisons with scikit-learn and broader dataset validation remain pending.

## Build and run Linear Regression

From the repository root, with a C++ compiler installed:

```bash
g++ -std=c++11 -Wall -Wextra -pedantic LinearRegression.cpp -o regression
./regression
```

Keep linear_regression.csv in the working directory. It contains two comma-separated numerical columns (x,y), no header, and at most 100 rows. Enter 7 at the prompt; the supplied sample gives approximately 13.2667, with training MSE 0.184127.

The earlier menu can be compiled separately:

```bash
g++ -std=c++11 ml_algorithms_menu.cpp -o ml_core
./ml_core
```

Only the regression dataset is currently included. The other menu choices require their datasets. The old menu expects one-column kmeans.csv; the planned standalone two-dimensional K-Means uses two columns. Align these formats when integrating.

Do not compile all .cpp files together yet: standalone programs have separate main() functions. Integration will introduce one entry point and reusable algorithm classes.

## File organization

| Location | Purpose |
|---|---|
| Repository root | C++ code, shared header and CSV datasets |
| reports/phase1/ | Phase 1 report source and PDF |
| reports/phase2/ | Phase 2 report PDF |
| reports/phase3/ | Place Phase 3 report files here when ready |
| presentations/ | Presentation source and PDF files |

## Team workflow

1. Pull the latest main before starting new work.
2. Create an algorithm branch: kmeans, knn, svm, hierarchical-clustering or pca. Use integration for connecting the completed classes to the menu.
3. Add the algorithm source and CSV. Reuse MathUtils.h and add new helpers inside namespace MathUtils; do not duplicate existing helpers.
4. Compile and run the changed program, then stage only the intended source/data/document files.
5. Commit with your own Git name and an email linked to your GitHub account. Push your branch and open a pull request into main.
6. Review and merge, then have teammates pull the updated main. Coordinate edits to MathUtils.h to avoid conflicts.

Use inline for functions defined in the shared header, or move their definitions into a MathUtils.cpp file when restructuring. This prevents duplicate-definition linker errors when several source files include the header.

## Team

- Shubhangani Kotiyal
- Ishanaya Agarwal
- Harshita Joshi

Each member should commit their own contributions. Branch names organize work; Git author identity determines commit attribution.
