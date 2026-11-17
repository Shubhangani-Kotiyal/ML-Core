#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <cassert>

// Arrays must contain finite numbers and have the stated lengths.
// assert checks invalid function arguments while developing the project.
// inline allows multiple .cpp files to include this header safely.
namespace MathUtils {

// ---------- Common calculations ----------

inline double mean(const double values[], int n) {
    assert(n > 0);
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += values[i];
    return sum / n;
}

inline double dotProduct(const double a[], const double b[], int n) {
    assert(n > 0);
    double total = 0;
    for (int i = 0; i < n; i++)
        total += a[i] * b[i];
    return total;
}

// The squared-distance formula is written only here.
inline double squaredDistance(const double a[], const double b[], int n) {
    assert(n > 0);
    double total = 0;
    for (int i = 0; i < n; i++) {
        double difference = a[i] - b[i];
        total += difference * difference;
    }
    return total;
}

// Overload for existing 2D code: delegates to the array version.
inline double squaredDistance(double x1, double y1, double x2, double y2) {
    double a[2] = {x1, y1};
    double b[2] = {x2, y2};
    return squaredDistance(a, b, 2);
}

// KNN can call this to display actual distances.
inline double euclideanDistance(double x1, double y1, double x2, double y2) {
    return std::sqrt(squaredDistance(x1, y1, x2, y2));
}

// MSE reuses the sum of squared differences.
inline double meanSquaredError(const double actual[],
                               const double predicted[], int n) {
    assert(n > 0);
    return squaredDistance(actual, predicted, n) / n;
}

// Sample covariance, shared by regression and PCA.
inline double covariance(const double x[], const double y[], int n) {
    assert(n >= 2);
    double meanX = mean(x, n);
    double meanY = mean(y, n);
    double total = 0;
    for (int i = 0; i < n; i++)
        total += (x[i] - meanX) * (y[i] - meanY);
    return total / (n - 1);
}

inline double variance(const double values[], int n) {
    return covariance(values, values, n);
}

// Shared weighted sum plus bias.
inline double linearScore(const double features[], const double weights[],
                          int n, double bias) {
    return dotProduct(features, weights, n) + bias;
}

// ---------- Linear Regression ----------

inline bool fitLine(const double x[], const double y[], int n,
                    double &slope, double &intercept) {
    if (n < 2)
        return false;
    double varX = variance(x, n);
    if (varX == 0)
        return false;

    slope = covariance(x, y, n) / varX;
    intercept = mean(y, n) - slope * mean(x, n);
    return true;
}

inline double predictLine(double x, double slope, double intercept) {
    double features[1] = {x};
    double weights[1] = {slope};
    return linearScore(features, weights, 1, intercept);
}

// ---------- K-Means ----------
// Update each centroid using mean() on its cluster's coordinates.

inline int nearestCentroid(double x, double y, const double centerX[],
                           const double centerY[], int k) {
    assert(k > 0);
    int nearest = 0;
    double minimum = squaredDistance(x, y, centerX[0], centerY[0]);
    for (int i = 1; i < k; i++) {
        double distance = squaredDistance(x, y, centerX[i], centerY[i]);
        if (distance < minimum) {
            minimum = distance;
            nearest = i;
        }
    }
    return nearest;
}

// ---------- Hierarchical Clustering ----------
// Member arrays contain valid indices of points in two nonempty clusters.

inline double singleLinkDistance(const double x[], const double y[],
                                 const int membersA[], int countA,
                                 const int membersB[], int countB) {
    assert(countA > 0 && countB > 0);
    int bestA = membersA[0];
    int bestB = membersB[0];
    double minimum = squaredDistance(x[bestA], y[bestA], x[bestB], y[bestB]);

    for (int i = 0; i < countA; i++) {
        for (int j = 0; j < countB; j++) {
            int a = membersA[i];
            int b = membersB[j];
            double distance = squaredDistance(x[a], y[a], x[b], y[b]);
            if (distance < minimum) {
                minimum = distance;
                bestA = a;
                bestB = b;
            }
        }
    }
    return euclideanDistance(x[bestA], y[bestA], x[bestB], y[bestB]);
}

// ---------- Linear SVM ----------

inline double svmScore(double x, double y,
                       const double weights[], double bias) {
    double features[2] = {x, y};
    return linearScore(features, weights, 2, bias);
}

// Hinge-loss update with L2 regularization.
inline void updateSVM(double x, double y, int label, double weights[],
                      double &bias, double learningRate, double lambda) {
    assert((label == -1 || label == 1) && learningRate > 0 && lambda >= 0);
    double features[2] = {x, y};
    bool insideMargin = label * svmScore(x, y, weights, bias) < 1;

    for (int j = 0; j < 2; j++) {
        double gradient = lambda * weights[j];
        if (insideMargin)
            gradient -= label * features[j];
        weights[j] -= learningRate * gradient;
    }
    if (insideMargin)
        bias += learningRate * label;
}

// ---------- PCA: exactly two features reduced to one ----------

inline void covarianceMatrix2D(const double x[], const double y[], int n,
                               double matrix[2][2]) {
    matrix[0][0] = variance(x, n);
    matrix[0][1] = covariance(x, y, n);
    matrix[1][0] = matrix[0][1];
    matrix[1][1] = variance(y, n);
}

// Eigenvalues of a symmetric 2x2 covariance matrix, largest first.
inline void eigenvalues2D(const double matrix[2][2], double eigenvalues[]) {
    double halfTrace = (matrix[0][0] + matrix[1][1]) / 2;
    double halfDifference = (matrix[0][0] - matrix[1][1]) / 2;
    double radius = std::hypot(halfDifference, matrix[0][1]);
    eigenvalues[0] = halfTrace + radius;
    eigenvalues[1] = halfTrace - radius;
}

inline bool principalComponent2D(const double matrix[2][2],
                                 double component[], double &explainedRatio) {
    double totalVariance = matrix[0][0] + matrix[1][1];
    if (totalVariance <= 0) {
        component[0] = 1;
        component[1] = 0;
        explainedRatio = 0;
        return false;
    }

    double eigenvalues[2];
    eigenvalues2D(matrix, eigenvalues);
    double angle = 0.5 * std::atan2(2 * matrix[0][1],
                                   matrix[0][0] - matrix[1][1]);
    component[0] = std::cos(angle);
    component[1] = std::sin(angle);
    explainedRatio = eigenvalues[0] / totalVariance;
    return true;
}

// The returned component is a unit vector. Equal eigenvalues mean
// no unique preferred direction; this implementation selects one.
inline bool fitPCA2D(const double x[], const double y[], int n,
                     double center[], double component[], double &explainedRatio) {
    if (n < 2)
        return false;
    center[0] = mean(x, n);
    center[1] = mean(y, n);
    double matrix[2][2];
    covarianceMatrix2D(x, y, n, matrix);
    return principalComponent2D(matrix, component, explainedRatio);
}

inline double projectPoint2D(double x, double y, const double center[],
                             const double component[]) {
    double centered[2] = {x - center[0], y - center[1]};
    return dotProduct(centered, component, 2);
}

} // namespace MathUtils

#endif

