"""Reference menu: run with python3 python/ml_core_python.py."""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

DATASETS = Path(__file__).resolve().parent.parent / "datasets"


def load_data(filename, columns):
    data = np.loadtxt(DATASETS / filename, delimiter=",", ndmin=2)
    if data.size == 0 or data.shape[1] != columns:
        raise ValueError(f"{filename}: expected nonempty data with {columns} columns.")
    if not np.isfinite(data).all():
        raise ValueError("Use finite numbers only.")
    return data


def read_count(prompt, maximum):
    value = int(input(prompt))
    if not 1 <= value <= maximum:
        raise ValueError(f"Enter an integer from 1 to {maximum}.")
    return value


def read_point():
    point = np.array([float(v) for v in input("Enter two features: ").split()])
    if point.size != 2 or not np.isfinite(point).all():
        raise ValueError("Enter two finite numbers separated by a space.")
    return point.reshape(1, 2)


def run_linear_regression():
    data = load_data("linear_regression.csv", 2)
    x, y = data[:, :1], data[:, 1]
    if len(data) < 2 or np.all(x == x[0]):
        raise ValueError("Need at least two points and variation in x.")
    model = LinearRegression().fit(x, y)
    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Training MSE:", mean_squared_error(y, model.predict(x)))
    value = float(input("Enter x to predict y: "))
    if not np.isfinite(value):
        raise ValueError("Enter a finite number.")
    print("Predicted y:", model.predict([[value]])[0])


def show_clusters(data, labels, count):
    for cluster in range(count):
        print(f"Cluster {cluster + 1}:")
        for index, point in enumerate(data):
            if labels[index] == cluster:
                print(f"P{index + 1}: {point}")


def run_kmeans():
    data = load_data("kmeans.csv", 2)
    k = read_count("Enter number of clusters: ", min(10, len(data)))
    model = KMeans(n_clusters=k, init=data[:k].copy(),
                   n_init=1, max_iter=100, random_state=0)
    labels = model.fit_predict(data)
    print("Centroids:", model.cluster_centers_)
    show_clusters(data, labels, k)
    print("Within-cluster sum of squared distances:", model.inertia_)


def run_svm():
    data = load_data("svm.csv", 3)
    x, y = data[:, :2], data[:, 2]
    if not np.isin(y, [-1, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("SVM requires both -1 and 1 labels only.")
    model = SVC(kernel="linear", C=1.0).fit(x, y)
    print("Weights:", model.coef_[0], "Bias:", model.intercept_[0])
    print("Training accuracy:", accuracy_score(y, model.predict(x)))
    print("Predicted class:", int(model.predict(read_point())[0]))


def run_hierarchical_clustering():
    data = load_data("hierarchical_clustering.csv", 2)
    target = read_count("Enter desired number of clusters: ", len(data))
    if len(data) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=target, linkage="single").fit_predict(data)
    show_clusters(data, labels, target)


def run_pca():
    data = load_data("pca.csv", 2)
    if len(data) < 2 or np.all(data == data[0]):
        raise ValueError("PCA needs at least two points and variation.")
    model = PCA(n_components=1, svd_solver="full")
    reduced = model.fit_transform(data)
    print("Means:", model.mean_)
    print("Principal component:", model.components_[0])
    print("Explained variance ratio:", model.explained_variance_ratio_[0])
    for point, result in zip(data, reduced[:, 0]):
        print(point, "->", result)


def run_knn():
    data = load_data("knn.csv", 3)
    x, y = data[:, :2], data[:, 2]
    if not np.equal(y, np.floor(y)).all():
        raise ValueError("KNN labels must be whole numbers.")
    k = read_count("Enter K: ", len(data))
    model = KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                metric="euclidean", algorithm="brute").fit(x, y)
    point = read_point()
    distances, indices = model.kneighbors(point)
    for distance, index in zip(distances[0], indices[0]):
        print(x[index], "Class:", int(y[index]), "Distance:", distance)
    print("Predicted class:", int(model.predict(point)[0]))


def main():
    functions = {
        "1": run_linear_regression, "2": run_kmeans, "3": run_svm,
        "4": run_hierarchical_clustering, "5": run_pca, "6": run_knn
    }
    while True:
        print("\nML-CORE PYTHON\n1. Linear Regression\n2. K-Means\n3. SVM")
        print("4. Hierarchical Clustering\n5. PCA\n6. KNN\n0. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "0":
            break
        if choice not in functions:
            print("Choose a number from 0 to 6.")
            continue
        try:
            functions[choice]()
        except (ValueError, OSError) as error:
            print("Error:", error)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting ML-Core.")
