# Contributing to ML-Core

## Branches

| Branch | Purpose |
|---|---|
| main | Reviewed, working project |
| linear-regression | Regression class and dataset |
| kmeans | K-Means class and dataset |
| svm | SVM class and dataset |
| hierarchical-clustering | Hierarchical class and dataset |
| pca | PCA class and dataset |
| knn | KNN class and dataset |
| math-utils | Shared mathematical functions |
| integration | Menu and changes affecting multiple algorithms |
| documentation | Reports, presentations and instructions |

Each branch contains the whole project so it can be compiled. Coordinate shared include/MathUtils.h edits with teammates.

## Start work

Save files and check git status before switching. Commit or stash unfinished changes first.

Example for KNN:

```bash
git fetch origin
git switch knn
git pull --ff-only origin knn
git merge origin/main
```

If the local branch does not exist and Git cannot infer it, use git switch --track origin/knn. Replace knn with the relevant branch name. Resolve merge conflicts before continuing.

## Test and commit

Build with make and run ./ml_core. Exercise the modified algorithm and other algorithms affected by shared functions.

```bash
git add src/knn.cpp datasets/knn.csv
git commit -m "Improve KNN input validation"
git push origin knn
```

Stage only intended source, dataset or documentation files. Open a pull request from the working branch into main and ask a teammate to review. Use Create a merge commit to retain the individual commits in the merged history.

Members should use their own Git name and an email linked to their GitHub account. Branch names organize work; they do not set commit authorship. College assessment rules may differ from GitHub contribution displays.

## Code conventions

- Keep one main() in src/ml_algorithms_menu.cpp and preserve the base-class execute interface and object accessor names.
- Reuse include/MathUtils.h rather than repeating formulas.
- Keep fixed-size arrays within bounds.
- Keep CSV filenames consistent with loadData calls.
- Do not upload compiled executables.
- Document algorithm limitations and distinguish training results from held-out evaluation.

Use task branches while work is active. After a pull request is merged, its branch can be deleted when nobody is using it. Existing algorithm branches may still show the previous layout; merge origin/main before editing. Do not delete branches containing unmerged work.
