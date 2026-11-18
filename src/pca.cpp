#include <iostream>
#include <fstream>
#include <cmath>
#include "MathUtils.h"
using namespace std;

class PCA {
private:
    double x[100], y[100];
    double transformed[100];
    int n;

    double center[2];
    double component[2];
    double explainedRatio;

public:
    PCA() {
        n = 0;
        center[0] = 0;
        center[1] = 0;
        component[0] = 0;
        component[1] = 0;
        explainedRatio = 0;
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
                cout << "Maximum 100 rows allowed.\n";
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

        if (n < 2) {
            cout << "At least two rows are required.\n";
            return false;
        }

        return true;
    }

    bool fit() {
        return MathUtils::fitPCA2D(
            x, y, n, center, component, explainedRatio
        );
    }

    void transform() {
        for (int i = 0; i < n; i++) {
            transformed[i] = MathUtils::projectPoint2D(
                x[i], y[i], center, component
            );
        }
    }

    void display() const {
        cout << "\nMean of first feature: " << center[0];
        cout << "\nMean of second feature: " << center[1];

        cout << "\n\nFirst principal component: (";
        cout << component[0] << ", " << component[1] << ")\n";

        cout << "Explained variance ratio: ";
        cout << explainedRatio << "\n";

        cout << "\nOriginal point -> Reduced value\n";

        for (int i = 0; i < n; i++) {
            cout << "(" << x[i] << ", " << y[i] << ")";
            cout << " -> " << transformed[i] << "\n";
        }
    }
};

int runPCA() {
    PCA model;

    if (!model.loadData("datasets/pca.csv"))
        return 1;

    if (!model.fit()) {
        cout << "PCA needs variation in the data.\n";
        return 1;
    }

    model.transform();
    model.display();

    return 0;
}