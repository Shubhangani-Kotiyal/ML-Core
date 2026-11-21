#ifndef ML_ALGORITHM_H
#define ML_ALGORITHM_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <climits>

class MLAlgorithm {
protected:
    enum { MAX_ROWS = 100 };
    double x[MAX_ROWS], y[MAX_ROWS];
    int labels[MAX_ROWS];
    int n;

    // labelMode: 0 = two numeric columns, 1 = SVM, 2 = KNN.
    bool loadCSV(const char filename[], int minimumRows, int labelMode = 0) {
        n = 0;
        std::ifstream file(filename);
        if (!file) {
            std::cout << "Unable to open CSV: " << filename << "\n";
            return false;
        }
        std::string line;
        int lineNumber = 0;
        int count = 0;
        while (std::getline(file, line)) {
            ++lineNumber;
            if (line.find_first_not_of(" \t\r") == std::string::npos)
                continue;
            if (count == MAX_ROWS) {
                std::cout << "Maximum 100 data rows allowed. File rejected.\n";
                return false;
            }
            std::stringstream row(line);
            double values[3] = {0, 0, 0};
            int columns = labelMode == 0 ? 2 : 3;
            for (int j = 0; j < columns; ++j) {
                if (!(row >> values[j]) || !std::isfinite(values[j])) {
                    std::cout << "Invalid or missing number on CSV line "
                              << lineNumber << ".\n";
                    return false;
                }
                if (j < columns - 1) {
                    char comma;
                    if (!(row >> comma) || comma != ',') {
                        std::cout << "Expected comma on CSV line "
                                  << lineNumber << ".\n";
                        return false;
                    }
                }
            }
            row >> std::ws;
            if (!row.eof()) {
                std::cout << "Extra data on CSV line " << lineNumber << ".\n";
                return false;
            }
            if (labelMode != 0) {
                double label = values[2];
                if (std::floor(label) != label || label < INT_MIN || label > INT_MAX) {
                    std::cout << "Class labels must be integers in the int range.\n";
                    return false;
                }
                if (labelMode == 1 && label != -1 && label != 1) {
                    std::cout << "SVM labels must be -1 or 1.\n";
                    return false;
                }
                labels[count] = static_cast<int>(label);
            }
            x[count] = values[0];
            y[count] = values[1];
            ++count;
        }
        if (file.bad()) {
            std::cout << "Error reading CSV file.\n";
            return false;
        }
        if (count < minimumRows) {
            std::cout << "At least " << minimumRows << " data rows required.\n";
            return false;
        }
        if (labelMode == 1) {
            bool positive = false, negative = false;
            for (int i = 0; i < count; ++i) {
                if (labels[i] == 1) positive = true;
                if (labels[i] == -1) negative = true;
            }
            if (!positive || !negative) {
                std::cout << "SVM data must contain both classes.\n";
                return false;
            }
        }
        n = count;
        return true;
    }

public:
    MLAlgorithm() : n(0) {}
    virtual ~MLAlgorithm() {}
    virtual int execute(const char filename[]) = 0;
};

// Objects remain in their own source files; the menu uses base references.
MLAlgorithm& getLinearRegression();
MLAlgorithm& getKMeans();
MLAlgorithm& getSVM();
MLAlgorithm& getHierarchicalClustering();
MLAlgorithm& getPCA();
MLAlgorithm& getKNN();

#endif
