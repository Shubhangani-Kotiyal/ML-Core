#include <iostream>
#include "MathUtils.h"
#include "MLAlgorithm.h"
using namespace std;

class LinearRegression : public MLAlgorithm {
private:
    double slope, intercept;

public:
    LinearRegression() {
        slope = 0;
        intercept = 0;
    }

    bool fit() {
        return MathUtils::fitLine(x, y, n, slope, intercept);
    }

    double predict(double value) const {
        return MathUtils::predictLine(value, slope, intercept);
    }

    double calculateMSE() const {
        double predicted[100];

        for (int i = 0; i < n; i++) {
            predicted[i] = predict(x[i]);
        }

        return MathUtils::meanSquaredError(y, predicted, n);
    }

    void display() const {
        cout << "\nSlope: " << slope;
        cout << "\nIntercept: " << intercept;
        cout << "\nTraining MSE: " << calculateMSE() << "\n";
    }
    int execute(const char filename[]) override {
        if (!loadCSV(filename, 2, 0))
            return 1;
    
        if (!fit()) {
            cout << "Cannot fit a line: all x values are equal.\n";
            return 1;
        }
    
        display();
    
        double value;
        cout << "\nEnter x to predict y: ";
    
        if (!(cin >> value) || !isfinite(value)) {
            cout << "Please enter a number.\n";
            return 1;
        }
    
        cout << "Predicted y: " << predict(value) << "\n";
    
        return 0;
    }
};

MLAlgorithm& getLinearRegression() {
    static LinearRegression model;
    return model;
}
