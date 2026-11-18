#include <iostream>
using namespace std;

int runLinearRegression();
int runKMeans();
int runSVM();
int runHierarchicalClustering();
int runPCA();
int runKNN();

int main() {
    int choice;
    do {
        cout << "\n========== ML-CORE ==========\n";
        cout << "1. Linear Regression\n2. K-Means Clustering\n";
        cout << "3. Support Vector Machine\n4. Hierarchical Clustering\n";
        cout << "5. Principal Component Analysis\n6. K-Nearest Neighbours\n";
        cout << "0. Exit\nEnter your choice: ";
        if (!(cin >> choice)) {
            cout << "Invalid input. Please restart and enter a number.\n";
            return 1;
        }
        int status = 0;
        switch (choice) {
            case 1: status = runLinearRegression(); break;
            case 2: status = runKMeans(); break;
            case 3: status = runSVM(); break;
            case 4: status = runHierarchicalClustering(); break;
            case 5: status = runPCA(); break;
            case 6: status = runKNN(); break;
            case 0: cout << "Exiting ML-Core.\n"; break;
            default: cout << "Please choose a number from 0 to 6.\n";
        }
        if (!cin) {
            cout << "Input stopped. Please restart the program.\n";
            return 1;
        }
        if (status != 0)
            cout << "The algorithm could not complete. Check the message above.\n";
    } while (choice != 0);
    return 0;
}
