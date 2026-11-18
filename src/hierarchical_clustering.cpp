#include <iostream>
#include <fstream>
#include <cmath>
#include "MathUtils.h"
using namespace std;

class HierarchicalClustering {
private:
    double x[100], y[100];
    int n;

    // Each row stores the point indices belonging to a cluster.
    int members[100][100];
    int count[100];
    bool active[100];
    int clusterCount;

public:
    HierarchicalClustering() {
        n = 0;
        clusterCount = 0;

        for (int i = 0; i < 100; i++) {
            count[i] = 0;
            active[i] = false;
        }
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

            double row[2];

            for (int j = 0; j < 2; j++) {
                while (file.peek() == ' ' ||
                       file.peek() == '\t') {
                    file.get();
                }

                if (file.peek() == '\n' ||
                    file.peek() == '\r' ||
                    !(file >> row[j]) ||
                    !isfinite(row[j])) {
                    cout << "Invalid CSV: use two numbers per row.\n";
                    return false;
                }

                while (file.peek() == ' ' ||
                       file.peek() == '\t') {
                    file.get();
                }

                if (j == 0) {
                    char comma;
                    file.get(comma);

                    if (!file || comma != ',') {
                        cout << "Expected a comma between values.\n";
                        return false;
                    }
                }
            }

            if (!file.eof() &&
                file.peek() != '\n' &&
                file.peek() != '\r') {
                cout << "Unexpected characters after the second value.\n";
                return false;
            }

            x[n] = row[0];
            y[n] = row[1];
            n++;
        }

        if (n == 0) {
            cout << "The dataset is empty.\n";
            return false;
        }

        return true;
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
};

int runHierarchicalClustering() {
    HierarchicalClustering model;

    if (!model.loadData("datasets/hierarchical_clustering.csv"))
        return 1;

    int target;

    cout << "Enter the desired number of clusters: ";

    if (!(cin >> target)) {
        cout << "Please enter an integer.\n";
        return 1;
    }

    if (!model.fit(target))
        return 1;

    model.display();

    return 0;
}