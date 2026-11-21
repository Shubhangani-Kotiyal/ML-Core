#include <iostream>
#include <cmath>
#include "MathUtils.h"
#include "MLAlgorithm.h"
using namespace std;

class PCA : public MLAlgorithm {
private:
    double transformed[100];

    double center[2];
    double component[2];
    double explainedRatio;

public:
    PCA() {
        center[0] = 0;
        center[1] = 0;
        component[0] = 0;
        component[1] = 0;
        explainedRatio = 0;
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
    int execute(const char filename[]) override {
        if (!loadCSV(filename, 2, 0))
            return 1;
    
        if (!fit()) {
            cout << "PCA needs variation in the data.\n";
            return 1;
        }
    
        transform();
        display();
    
        return 0;
    }
};

MLAlgorithm& getPCA() {
    static PCA model;
    return model;
}
