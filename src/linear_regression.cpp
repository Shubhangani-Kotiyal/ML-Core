#include <iostream>
#include <fstream>
#include "MathUtils.h"
using namespace std;

class LinearRegression {
private:
    double x[100], y[100];
    int n;
    double slope, intercept;

public:
    LinearRegression() {
        n = 0;
        slope = 0;
        intercept = 0;
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
                cout << "Maximum 100 data rows allowed.\n";
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

        if (n < 2) {
            cout << "At least two data rows are required.\n";
            return false;
        }

        return true;
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
};

int runLinearRegression() {
    LinearRegression model;

    if (!model.loadData("datasets/linear_regression.csv"))
        return 1;

    if (!model.fit()) {
        cout << "Cannot fit a line: all x values are equal.\n";
        return 1;
    }

    model.display();

    double value;
    cout << "\nEnter x to predict y: ";

    if (!(cin >> value)) {
        cout << "Please enter a number.\n";
        return 1;
    }

    cout << "Predicted y: " << model.predict(value) << "\n";

    return 0;
}