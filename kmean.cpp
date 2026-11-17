#include <iostream>
#include <fstream>
#include "MathUtils.h"
using namespace std;

class KMeans {
private:
    double x[100], y[100];
    double centerX[10], centerY[10];
    int cluster[100];
    int n, k;

public:
    KMeans() {
        n = 0;
        k = 0;
    }

    bool loadData(const char filename[]) {
        ifstream file(filename);

        if (!file) {
            cout << "Unable to open the CSV file.\n";
            return false;
        }

        n = 0;

        while (true) {
            file >> ws;

            if (file.eof())
                break;

            if (n == 100) {
                cout << "Maximum 100 points allowed.\n";
                return false;
            }

            char comma;

            if (!(file >> x[n])) {
                cout << "Invalid x value.\n";
                return false;
            }

            file.get(comma);

            if (comma != ',' || !(file >> y[n])) {
                cout << "Each row must contain two numbers: x,y\n";
                return false;
            }

            while (file.peek() == ' ' || file.peek() == '\t')
                file.get();

            if (file.peek() != '\n' &&
                file.peek() != '\r' &&
                !file.eof()) {
                cout << "Unexpected data after y.\n";
                return false;
            }

            n++;
        }

        if (n == 0) {
            cout << "The dataset is empty.\n";
            return false;
        }

        return true;
    }

    bool fit(int numberOfClusters) {
        if (numberOfClusters < 1 ||
            numberOfClusters > 10 ||
            numberOfClusters > n) {
            cout << "K must be between 1 and 10";
            cout << " and cannot exceed the number of points.\n";
            return false;
        }

        k = numberOfClusters;

        // Choose the first K points as initial centroids.
        for (int i = 0; i < k; i++) {
            centerX[i] = x[i];
            centerY[i] = y[i];
        }

        for (int i = 0; i < n; i++) {
            cluster[i] = -1;
        }

        // Repeat assignment and centroid updates.
        for (int iteration = 0; iteration < 100; iteration++) {
            bool changed = false;

            // Assign each point to its nearest centroid.
            for (int i = 0; i < n; i++) {
                int nearest = MathUtils::nearestCentroid(
                    x[i], y[i], centerX, centerY, k
                );

                if (cluster[i] != nearest) {
                    cluster[i] = nearest;
                    changed = true;
                }
            }

            // Stop when the assignments no longer change.
            if (!changed)
                return true;

            // Recalculate each centroid using its cluster's points.
            for (int c = 0; c < k; c++) {
                double groupX[100], groupY[100];
                int count = 0;

                for (int i = 0; i < n; i++) {
                    if (cluster[i] == c) {
                        groupX[count] = x[i];
                        groupY[count] = y[i];
                        count++;
                    }
                }

                // Keep the previous centroid if a cluster is empty.
                if (count > 0) {
                    centerX[c] = MathUtils::mean(groupX, count);
                    centerY[c] = MathUtils::mean(groupY, count);
                }
            }
        }

        cout << "Stopped after 100 iterations.\n";
        return true;
    }

    void display() const {
        cout << "\nK-Means Results\n";

        for (int c = 0; c < k; c++) {
            cout << "\nCluster " << c + 1;
            cout << "\nCentroid: (" << centerX[c];
            cout << ", " << centerY[c] << ")\n";

            for (int i = 0; i < n; i++) {
                if (cluster[i] == c) {
                    cout << "(" << x[i] << ", " << y[i] << ")\n";
                }
            }
        }
    }
};

int runKMeans() {
    KMeans model;

    if (!model.loadData("kmeans.csv"))
        return 1;

    int k;

    cout << "Enter the number of clusters: ";

    if (!(cin >> k)) {
        cout << "Please enter an integer.\n";
        return 1;
    }

    if (!model.fit(k))
        return 1;

    model.display();

    return 0;
}