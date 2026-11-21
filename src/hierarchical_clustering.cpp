#include <iostream>
#include <cmath>
#include "MathUtils.h"
#include "MLAlgorithm.h"
using namespace std;

class HierarchicalClustering : public MLAlgorithm {
private:

    // Each row stores the point indices belonging to a cluster.
    int members[100][100];
    int count[100];
    bool active[100];
    int clusterCount;

public:
    HierarchicalClustering() {
        clusterCount = 0;

        for (int i = 0; i < 100; i++) {
            count[i] = 0;
            active[i] = false;
        }
    }

    bool fit(int targetClusters) {
        if (targetClusters < 1 || targetClusters > n) {
            cout << "Cluster count must be between 1 and " << n << ".\n";
            return false;
        }

        // Initially, every point forms its own cluster.
        for (int i = 0; i < n; i++) {
            members[i][0] = i;
            count[i] = 1;
            active[i] = true;
        }

        clusterCount = n;

        while (clusterCount > targetClusters) {
            int bestA = -1;
            int bestB = -1;
            double minimumDistance = 0;

            // Find the closest pair of active clusters.
            for (int i = 0; i < n; i++) {
                if (!active[i])
                    continue;

                for (int j = i + 1; j < n; j++) {
                    if (!active[j])
                        continue;

                    double distance = MathUtils::singleLinkDistance(
                        x, y,
                        members[i], count[i],
                        members[j], count[j]
                    );

                    if (bestA == -1 || distance < minimumDistance) {
                        minimumDistance = distance;
                        bestA = i;
                        bestB = j;
                    }
                }
            }

            // Move every member of cluster B into cluster A.
            for (int i = 0; i < count[bestB]; i++) {
                members[bestA][count[bestA]] = members[bestB][i];
                count[bestA]++;
            }

            active[bestB] = false;
            count[bestB] = 0;
            clusterCount--;

            cout << "Merged group " << bestB + 1;
            cout << " into group " << bestA + 1;
            cout << " at distance " << minimumDistance << "\n";
        }

        return true;
    }

    void display() const {
        int displayNumber = 1;

        cout << "\nFinal clusters:\n";

        for (int i = 0; i < n; i++) {
            if (!active[i])
                continue;

            cout << "\nCluster " << displayNumber << ":\n";

            for (int j = 0; j < count[i]; j++) {
                int point = members[i][j];

                cout << "P" << point + 1;
                cout << " (" << x[point] << ", " << y[point] << ")\n";
            }

            displayNumber++;
        }
    }
    int execute(const char filename[]) override {
        if (!loadCSV(filename, 1, 0))
            return 1;
    
        int target;
    
        cout << "Enter the desired number of clusters: ";
    
        if (!(cin >> target)) {
            cout << "Please enter an integer.\n";
            return 1;
        }
    
        if (!fit(target))
            return 1;
    
        display();
    
        return 0;
    }
};

MLAlgorithm& getHierarchicalClustering() {
    static HierarchicalClustering model;
    return model;
}
