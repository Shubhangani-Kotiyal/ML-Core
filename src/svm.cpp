#include <iostream>
#include <cmath>
#include "MathUtils.h"
#include "MLAlgorithm.h"
using namespace std;

class SVM : public MLAlgorithm {
private:

    double weights[2];
    double bias;
    double learningRate;
    double lambda;
    int epochs;

public:
    SVM() {
        weights[0] = 0;
        weights[1] = 0;
        bias = 0;

        learningRate = 0.001;
        lambda = 0.01;
        epochs = 2000;
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
    int execute(const char filename[]) override {
        if (!loadCSV(filename, 1, 1))
            return 1;
    
        fit();
        display();
    
        double newX, newY;
    
        cout << "\nEnter two features of the new point: ";
    
        if (!(cin >> newX >> newY) ||
            !isfinite(newX) || !isfinite(newY)) {
            cout << "Please enter two finite numbers.\n";
            return 1;
        }
    
        cout << "Predicted class: ";
        cout << predict(newX, newY) << "\n";
    
        return 0;
    }
};

MLAlgorithm& getSVM() {
    static SVM model;
    return model;
}
