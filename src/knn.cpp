#include <iostream>
#include <cmath>
#include "MathUtils.h"
#include "MLAlgorithm.h"

using namespace std;

class KNN : public MLAlgorithm {
public:
    KNN() {
    }

    int getSize() const {
        return n;
    }

    // Call after loading data and checking that 1 <= k <= n.
    int predict(double newX, double newY, int k) const {
        double distance[100];
        int index[100];

        // Reuse the distance formula from MathUtils.h.
        for (int i = 0; i < n; i++) {
            distance[i] = MathUtils::euclideanDistance(
                newX, newY, x[i], y[i]
            );

            index[i] = i;
        }

        // Sort distances, keeping each point's index alongside it.
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (distance[j] > distance[j + 1]) {
                    double tempDistance = distance[j];
                    distance[j] = distance[j + 1];
                    distance[j + 1] = tempDistance;

                    int tempIndex = index[j];
                    index[j] = index[j + 1];
                    index[j + 1] = tempIndex;
                }
            }
        }

        cout << "\nNearest neighbours:\n";

        for (int i = 0; i < k; i++) {
            int point = index[i];

            cout << "(" << x[point] << ", " << y[point] << ")";
            cout << "  Class: " << labels[point];
            cout << "  Distance: " << distance[i] << "\n";
        }

        int predictedClass = labels[index[0]];
        int highestVotes = 0;

        // Count votes for each class among the nearest K points.
        for (int i = 0; i < k; i++) {
            int currentClass = labels[index[i]];
            int votes = 0;

            for (int j = 0; j < k; j++) {
                if (labels[index[j]] == currentClass)
                    votes++;
            }

            // If votes tie, choose the smaller class label.
            if (votes > highestVotes ||
                (votes == highestVotes &&
                 currentClass < predictedClass)) {
                highestVotes = votes;
                predictedClass = currentClass;
            }
        }

        return predictedClass;
    }

    int execute(const char filename[]) override {
        if (!loadCSV(filename, 1, 2))
            return 1;
    
        int k;
        double newX, newY;
    
        cout << "Enter K (1 to " << getSize() << "): ";
    
        if (!(cin >> k) || k < 1 || k > getSize()) {
            cout << "Invalid K.\n";
            return 1;
        }
    
        cout << "Enter the two features of the point: ";
    
        if (!(cin >> newX >> newY) ||
            !isfinite(newX) || !isfinite(newY)) {
            cout << "Please enter two valid numbers.\n";
            return 1;
        }
    
        int result = predict(newX, newY, k);
    
        cout << "\nPredicted class: " << result << "\n";
    
        return 0;
    }
};

MLAlgorithm& getKNN() {
    static KNN model;
    return model;
}
