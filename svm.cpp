#include <iostream>
#include <fstream>
#include <cmath>
#include "MathUtils.h"
using namespace std;

class SVM {
private:
    double x[100], y[100];
    int labels[100];
    int n;

    double weights[2];
    double bias;
    double learningRate;
    double lambda;
    int epochs;

public:
    SVM() {
        n = 0;
        weights[0] = 0;
        weights[1] = 0;
        bias = 0;

        learningRate = 0.001;
        lambda = 0.01;
        epochs = 2000;
    }

    bool loadData(const char filename[]) {
        ifstream file(filename);

        if (!file) {
            cout << "Unable to open the CSV file.\n";
            return false;
        }

        n = 0;
        int positive = 0, negative = 0;

        while (true) {
            file >> ws;

            if (file.eof())
                break;

            if (n == 100) {
                cout << "Maximum 100 rows allowed.\n";
                return false;
            }

            double row[3];

            for (int j = 0; j < 3; j++) {
                // Skip spaces without skipping a missing value's newline.
                while (file.peek() == ' ' || file.peek() == '\t')
                    file.get();

                if (file.peek() == '\n' || file.peek() == '\r' ||
                    !(file >> row[j]) || !isfinite(row[j])) {
                    cout << "Invalid CSV: use x,y,label with no header.\n";
                    return false;
                }

                while (file.peek() == ' ' || file.peek() == '\t')
                    file.get();

                if (j < 2) {
                    char comma;
                    file.get(comma);

                    if (!file || comma != ',') {
                        cout << "Expected a comma between values.\n";
                        return false;
                    }
                }
            }

            if (!file.eof() &&
                file.peek() != '\n' && file.peek() != '\r') {
                cout << "Unexpected characters after the label.\n";
                return false;
            }

            if (row[2] != -1 && row[2] != 1) {
                cout << "Labels must be -1 or 1.\n";
                return false;
            }

            x[n] = row[0];
            y[n] = row[1];
            labels[n] = (int)row[2];

            if (labels[n] == 1)
                positive++;
            else
                negative++;

            n++;
        }

        if (positive == 0 || negative == 0) {
            cout << "The dataset must contain both classes.\n";
            return false;
        }

        return true;
    }

    void fit() {
        // Start fresh whenever training is called.
        weights[0] = 0;
        weights[1] = 0;
        bias = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < n; i++) {
                MathUtils::updateSVM(
                    x[i], y[i], labels[i],
                    weights, bias, learningRate, lambda
                );
            }
        }
    }

    int predict(double newX, double newY) const {
        double score = MathUtils::svmScore(
            newX, newY, weights, bias
        );

        if (score >= 0)
            return 1;

        return -1;
    }

    void display() const {
        cout << "\nWeight 1: " << weights[0];
        cout << "\nWeight 2: " << weights[1];
        cout << "\nBias: " << bias << "\n";

        int correct = 0;

        cout << "\nTraining predictions:\n";

        for (int i = 0; i < n; i++) {
            int result = predict(x[i], y[i]);

            cout << "(" << x[i] << ", " << y[i] << ")";
            cout << " Actual: " << labels[i];
            cout << " Predicted: " << result << "\n";

            if (result == labels[i])
                correct++;
        }

        cout << "\nCorrect training predictions: ";
        cout << correct << " out of " << n << "\n";
    }
};

int runSVM() {
    SVM model;

    if (!model.loadData("svm.csv"))
        return 1;

    model.fit();
    model.display();

    double newX, newY;

    cout << "\nEnter two features of the new point: ";

    if (!(cin >> newX >> newY) ||
        !isfinite(newX) || !isfinite(newY)) {
        cout << "Please enter two finite numbers.\n";
        return 1;
    }

    cout << "Predicted class: ";
    cout << model.predict(newX, newY) << "\n";

    return 0;
}