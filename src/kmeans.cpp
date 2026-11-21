#include <iostream>
#include "MathUtils.h"
#include "MLAlgorithm.h"
using namespace std;

class KMeans : public MLAlgorithm {
private:
    double centerX[10], centerY[10];
    int cluster[100];
    int k;

public:
    KMeans() {
        k = 0;
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

        // Repeated rows must not create identical initial centroids.
        int selected = 0;
        for (int i = 0; i < n && selected < k; i++) {
            bool duplicate = false;
            for (int c = 0; c < selected; c++) {
                if (x[i] == centerX[c] && y[i] == centerY[c])
                    duplicate = true;
            }
            if (!duplicate) {
                centerX[selected] = x[i];
                centerY[selected] = y[i];
                selected++;
            }
        }
        if (selected < k) {
            cout << "K cannot exceed the number of distinct points.\n";
            return false;
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

            int counts[10] = {0};
            for (int i = 0; i < n; i++)
                counts[cluster[i]]++;

            // Fill an empty cluster without emptying its donor cluster.
            for (int c = 0; c < k; c++) {
                if (counts[c] != 0)
                    continue;
                int candidate = -1;
                double farthest = -1;
                for (int i = 0; i < n; i++) {
                    int donor = cluster[i];
                    if (counts[donor] <= 1)
                        continue;
                    double distance = MathUtils::squaredDistance(
                        x[i], y[i], centerX[donor], centerY[donor]);
                    if (distance > farthest) {
                        farthest = distance;
                        candidate = i;
                    }
                }
                if (candidate == -1) {
                    cout << "Unable to fill an empty cluster.\n";
                    return false;
                }
                counts[cluster[candidate]]--;
                cluster[candidate] = c;
                counts[c]++;
                changed = true;
            }

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

                centerX[c] = MathUtils::mean(groupX, count);
                centerY[c] = MathUtils::mean(groupY, count);
            }
        }

        cout << "Warning: reached 100 iterations; convergence is not guaranteed.\n";
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
    int execute(const char filename[]) override {
        if (!loadCSV(filename, 1, 0))
            return 1;
    
        int k;
    
        cout << "Enter the number of clusters: ";
    
        if (!(cin >> k)) {
            cout << "Please enter an integer.\n";
            return 1;
        }
    
        if (!fit(k))
            return 1;
    
        display();
    
        return 0;
    }
};

MLAlgorithm& getKMeans() {
    static KMeans model;
    return model;
}

