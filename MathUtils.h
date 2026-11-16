#ifndef MATH_UTILS_H
#define MATH_UTILS_H

namespace MathUtils {

    inline double mean(const double values[], int n) {
        double sum = 0;

        for (int i = 0; i < n; i++) {
            sum += values[i];
        }

        return sum / n;
    }

    inline bool fitLine(const double x[], const double y[], int n,
                 double &slope, double &intercept) {
        if (n < 2)
            return false;

        double meanX = mean(x, n);
        double meanY = mean(y, n);

        double numerator = 0;
        double denominator = 0;

        for (int i = 0; i < n; i++) {
            numerator += (x[i] - meanX) * (y[i] - meanY);
            denominator += (x[i] - meanX) * (x[i] - meanX);
        }

        if (denominator == 0)
            return false;

        slope = numerator / denominator;
        intercept = meanY - slope * meanX;

        return true;
    }

    inline double predictLine(double x, double slope, double intercept) {
        return slope * x + intercept;
    }

    inline double meanSquaredError(const double actual[],
                            const double predicted[], int n) {
        double totalError = 0;

        for (int i = 0; i < n; i++) {
            double error = actual[i] - predicted[i];
            totalError += error * error;
        }

        return totalError / n;
    }
// Squared distance between two points.
// K-Means can compare squared distances without using square roots.
inline double squaredDistance(double x1, double y1,
                              double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;

    return dx * dx + dy * dy;
}

// Find the closest centroid.
inline int nearestCentroid(double x, double y,
                           const double centerX[],
                           const double centerY[], int k) {
    int nearest = 0;

    double minimum = squaredDistance(
        x, y, centerX[0], centerY[0]
    );

    for (int i = 1; i < k; i++) {
        double distance = squaredDistance(
            x, y, centerX[i], centerY[i]
        );

        if (distance < minimum) {
            minimum = distance;
            nearest = i;
        }
    }

    return nearest;
}

} // namespace MathUtils

#endif
