#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "MathUtils.h"

using namespace std;

class KNN {
private:
    double x[100], y[100];
    int label[100];
    int n;

public:
    KNN() {
        n = 0;
    }

    bool loadData(const char filename[]) {
        ifstream file(filename);

        if (!file) {
            cout << "Unable to open the CSV file.\n";
            return false;
        }

        n = 0;
        string line;

        while (getline(file, line)) {
            // Skip empty lines.
            if (line.find_first_not_of(" \t\r") == string::npos)
                continue;

            if (n == 100) {
                cout << "Maximum 100 points allowed.\n";
                return false;
            }

            stringstream row(line);
            double first, second;
            int classLabel;
            char comma1, comma2;

            if (!(row >> first >> comma1 >> second
                      >> comma2 >> classLabel) ||
                comma1 != ',' || comma2 != ',' ||
                !isfinite(first) || !isfinite(second)) {
                cout << "Invalid CSV: expected x,y,integer_label.\n";
                return false;
            }

            // Reject extra columns or characters.
            row >> ws;

            if (!row.eof()) {
                cout << "Unexpected characters after the label.\n";
                return false;
            }

            x[n] = first;
            y[n] = second;
            label[n] = classLabel;
            n++;
        }

        if (n == 0) {
            cout << "The dataset is empty.\n";
            return false;
        }

        return true;
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
            cout << "  Class: " << label[point];
            cout << "  Distance: " << distance[i] << "\n";
        }

        int predictedClass = label[index[0]];
        int highestVotes = 0;

        // Count votes for each class among the nearest K points.
        for (int i = 0; i < k; i++) {
            int currentClass = label[index[i]];
            int votes = 0;

            for (int j = 0; j < k; j++) {
                if (label[index[j]] == currentClass)
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
};

int runKNN() {
    KNN model;

    if (!model.loadData("knn.csv"))
        return 1;

    int k;
    double newX, newY;

    cout << "Enter K (1 to " << model.getSize() << "): ";

    if (!(cin >> k) || k < 1 || k > model.getSize()) {
        cout << "Invalid K.\n";
        return 1;
    }

    cout << "Enter the two features of the point: ";

    if (!(cin >> newX >> newY) ||
        !isfinite(newX) || !isfinite(newY)) {
        cout << "Please enter two valid numbers.\n";
        return 1;
    }

    int result = model.predict(newX, newY, k);

    cout << "\nPredicted class: " << result << "\n";

    return 0;
}