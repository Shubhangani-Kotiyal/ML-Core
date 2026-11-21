#include <iostream>
#include <string>
#include "MLAlgorithm.h"
using namespace std;

int main() {
    MLAlgorithm* models[] = {
        &getLinearRegression(), &getKMeans(), &getSVM(),
        &getHierarchicalClustering(), &getPCA(), &getKNN()
    };
    const char* defaults[] = {
        "datasets/linear_regression.csv", "datasets/kmeans.csv",
        "datasets/svm.csv", "datasets/hierarchical_clustering.csv",
        "datasets/pca.csv", "datasets/knn.csv"
    };
    while (true) {
        cout << "\n========== ML-CORE ==========\n";
        cout << "1. Linear Regression\n2. K-Means\n3. SVM\n";
        cout << "4. Hierarchical Clustering\n5. PCA\n6. KNN\n0. Exit\n";
        cout << "Enter choice: ";
        int choice;
        if (!(cin >> choice)) {
            cout << "Invalid input. Restart and enter an integer.\n";
            return 1;
        }
        if (choice == 0) return 0;
        if (choice < 1 || choice > 6) {
            cout << "Choose 0 to 6.\n";
            continue;
        }
        cout << "CSV path (or - for " << defaults[choice - 1] << "): ";
        string filename;
        if (!getline(cin >> ws, filename)) return 1;
        if (filename == "-") filename = defaults[choice - 1];

        // Runtime polymorphism: calls the selected derived class's override.
        int status = models[choice - 1]->execute(filename.c_str());
        if (!cin) {
            cout << "Input stopped. Restart the program.\n";
            return 1;
        }
        if (status != 0) cout << "Algorithm could not complete; see message above.\n";
    }
}
