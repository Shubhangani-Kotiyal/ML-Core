# ⚡ ML-Core Visualizer — Interactive Web Frontend

An interactive, modern web frontend that visually demonstrates the 6 fundamental machine learning algorithms implemented from scratch in C++ for the **ML-Core** Project-Based Learning project at **Graphic Era Deemed to be University**.

> **Architecture & Implementation Clarification**: The web visualizer is a standalone client-side browser application. It executes independent JavaScript algorithms designed to mirror the numerical calculations and validation constraints of the C++ codebase (enforcing 100-row limits, distinct K-Means centroid selection, empty-cluster donor rebalancing, and zero-variance linear regression error handling). It runs completely client-side in the browser and does not call C++ binaries or WebAssembly.

---

## 👥 Team T170
- **Shubhangani Kotiyal** (Team Lead)
- **Ishanaya Agarwal**
- **Harshita Joshi**

---

## 🌟 Visualized Algorithms

1. **Linear Regression (Ordinary Least Squares)**
   - Visualizes training points, best-fit line ($y = mx + c$), and residual error segments (MSE).
   - Interactive probe $x \to \hat{y}$ (exact match to C++ benchmark: for $x=7$, $\hat{y} = 13.2667$, MSE: $0.184127$).
   - Click canvas to add custom points and watch parameters update dynamically in real time.

2. **K-Means Clustering (Lloyd's Iteration)**
   - Visualizes 2D coordinates colored by cluster and moving centroids ($\bigstar$).
   - Interactive stepping ("Next Step", "Auto Play", "Reset") demonstrating the alternation between point assignment and centroid recalculation.
   - Exact convergence to centroids `(1.75, 1.625)` and `(5.75, 5.625)`.

3. **Linear Support Vector Machine (Linear SVM)**
   - Visualizes the separating hyperplane ($w \cdot x + b = 0$), positive and negative margins ($w \cdot x + b = \pm 1$), and shaded margin corridor.
   - Highlights Support Vectors sitting along or within the margin boundary.
   - Draggable test query point with instant classification score calculation.

4. **Hierarchical Clustering (Agglomerative Single Linkage)**
   - Demonstrates bottom-up clustering with distance-based convex bubble enclosures.
   - Slider to select target clusters ($K$), dynamically cutting the dendrogram tree at the exact merge distances.

5. **Principal Component Analysis (PCA: 2D &rarr; 1D)**
   - Computes 2D mean center, sample covariance matrix, and eigenvalues.
   - Plots the primary principal component axis (PC1) capturing **96.32%** of total variance.
   - Toggle to project 2D points onto the 1D subspace with animated projection drop lines.

6. **K-Nearest Neighbors (KNN Classifier)**
   - Interactive draggable query probe $(x, y)$.
   - Expanding neighborhood radar radius highlighting the $K$ closest points.
   - Live voting breakdown and class assignment (e.g. at $(2, 2)$ with $K=3 \to$ Class 0).

---

## 🚀 How to Run Locally

### Option 1: Direct in Browser
Double-click `index.html` or open it with any modern web browser:
```bash
open index.html
```

### Option 2: Local HTTP Server (Python)
```bash
cd frontend
python3 -m http.server 8085
```
Then visit [http://localhost:8085](http://localhost:8085) in your browser.

---

## 🌐 Live Internet URL (Public Web Access)

Access the live deployed website instantly from any computer, phone, or browser worldwide:  
👉 **[https://stunning-taffy-3ed8d5.netlify.app](https://stunning-taffy-3ed8d5.netlify.app)**

---

## 📄 Deploy to GitHub Pages (Alternative Permanent Host)
To host this live on the web for presentations:
1. Go to your repository settings on GitHub.
2. Under **Pages**, select source branch `main` and folder `/frontend` (or copy `index.html`, `style.css`, and `app.js` to `gh-pages` branch).
3. Your live interactive website will be available at:
   `https://<username>.github.io/<repo-name>/frontend/`
