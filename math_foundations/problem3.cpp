#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;

// Structure to hold the three point sets from a data file
struct ScanData {
    MatrixXd points1;           // 2xN - points from pose 1
    MatrixXd points2_noiseless; // 2xN - same points from pose 2 (no noise)
    MatrixXd points2_noisy;     // 2xN - same points from pose 2 (with noise)
};

// Read data from comma-separated txt file
// Format: x1, y1, x2_noiseless, y2_noiseless, x2_noisy, y2_noisy
ScanData readDataFile(const string& filename) {
    ScanData data;
    vector<double> x1_vec, y1_vec;
    vector<double> x2_noiseless_vec, y2_noiseless_vec;
    vector<double> x2_noisy_vec, y2_noisy_vec;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        vector<double> values;

        // Parse comma-separated values
        while (getline(ss, token, ',')) {
            values.push_back(stod(token));
        }

        // Each line should have 6 values
        if (values.size() == 6) {
            x1_vec.push_back(values[0]);
            y1_vec.push_back(values[1]);
            x2_noiseless_vec.push_back(values[2]);
            y2_noiseless_vec.push_back(values[3]);
            x2_noisy_vec.push_back(values[4]);
            y2_noisy_vec.push_back(values[5]);
        }
    }
    file.close();

    int N = x1_vec.size();

    // Convert vectors to Eigen matrices (2xN format)
    data.points1.resize(2, N);
    data.points2_noiseless.resize(2, N);
    data.points2_noisy.resize(2, N);

    for (int i = 0; i < N; ++i) {
        data.points1(0, i) = x1_vec[i];
        data.points1(1, i) = y1_vec[i];
        data.points2_noiseless(0, i) = x2_noiseless_vec[i];
        data.points2_noiseless(1, i) = y2_noiseless_vec[i];
        data.points2_noisy(0, i) = x2_noisy_vec[i];
        data.points2_noisy(1, i) = y2_noisy_vec[i];
    }

    return data;
}

// SVD-based scan matching algorithm
// Returns 3x3 homogeneous transformation matrix T that transforms points2 to points1 frame
// T maps from frame 2 to frame 1: p1 = T * p2
Matrix3d scanMatch(const MatrixXd& points1, const MatrixXd& points2) {
    int N = points1.cols();

    // Compute centroids
    Vector2d centroid1 = points1.rowwise().mean();
    Vector2d centroid2 = points2.rowwise().mean();

    // Center the point sets
    MatrixXd centered1 = points1.colwise() - centroid1;
    MatrixXd centered2 = points2.colwise() - centroid2;

    // Compute cross-covariance matrix H = sum of (centered1_i * centered2_i^T)
    Matrix2d H = centered1 * centered2.transpose();

    // Perform SVD on H
    JacobiSVD<Matrix2d> svd(H, ComputeFullU | ComputeFullV);
    Matrix2d U = svd.matrixU();
    Matrix2d V = svd.matrixV();

    // Compute rotation matrix R = U * V^T
    Matrix2d R = U * V.transpose();

    // Ensure proper rotation (det(R) = 1, not -1 for reflection)
    if (R.determinant() < 0) {
        V.col(1) *= -1;
        R = U * V.transpose();
    }

    // Compute translation: t = centroid1 - R * centroid2
    Vector2d t = centroid1 - R * centroid2;

    // Build 3x3 homogeneous transformation matrix
    Matrix3d T = Matrix3d::Identity();
    T.block<2, 2>(0, 0) = R;
    T.block<2, 1>(0, 2) = t;

    return T;
}

// Apply homogeneous transformation to 2D points
MatrixXd transformPoints(const Matrix3d& T, const MatrixXd& points) {
    int N = points.cols();
    MatrixXd points_homo(3, N);

    // Convert to homogeneous coordinates
    points_homo.topRows(2) = points;
    points_homo.row(2) = VectorXd::Ones(N);

    // Apply transformation
    MatrixXd transformed_homo = T * points_homo;

    // Convert back to 2D
    return transformed_homo.topRows(2);
}

// Save points to CSV file
void savePointsToCSV(const MatrixXd& points, const string& filename) {
    ofstream file(filename);
    file << "x,y\n";
    for (int i = 0; i < points.cols(); ++i) {
        file << points(0, i) << "," << points(1, i) << "\n";
    }
    file.close();
}

int main() {
    cout << "=== Problem 3: Scan Matching ===\n\n";

    // Open CSV file for transformation matrices
    ofstream results_file("scan_results.csv");
    results_file << "scan_number,type,T00,T01,T02,T10,T11,T12,T20,T21,T22\n";

    // Process all 5 data files
    for (int file_idx = 1; file_idx <= 5; ++file_idx) {
        string filename = "../ScanMatchingData_" + to_string(file_idx) + ".txt";

        cout << "Processing " << filename << "...\n";
        cout << string(60, '-') << "\n";

        // Read data
        ScanData data = readDataFile(filename);

        if (data.points1.cols() == 0) {
            cout << "Failed to read data from " << filename << "\n\n";
            continue;
        }

        int N = data.points1.cols();
        cout << "Loaded " << N << " point correspondences\n\n";

        // Scan match with noiseless data
        Matrix3d T_noiseless = scanMatch(data.points1, data.points2_noiseless);

        cout << "Transformation (noiseless data):\n";
        cout << T_noiseless << "\n\n";

        // Save noiseless transformation to CSV
        results_file << file_idx << ",noiseless";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                results_file << "," << T_noiseless(i, j);
            }
        }
        results_file << "\n";

        // Scan match with noisy data
        Matrix3d T_noisy = scanMatch(data.points1, data.points2_noisy);

        cout << "Transformation (noisy data):\n";
        cout << T_noisy << "\n\n";

        // Save noisy transformation to CSV
        results_file << file_idx << ",noisy";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                results_file << "," << T_noisy(i, j);
            }
        }
        results_file << "\n";

        // Transform points2 back to frame 1 for visualization
        MatrixXd points2_noiseless_transformed = transformPoints(T_noiseless, data.points2_noiseless);
        MatrixXd points2_noisy_transformed = transformPoints(T_noisy, data.points2_noisy);

        // Save to CSV files for plotting
        string prefix = "scan_" + to_string(file_idx);
        savePointsToCSV(data.points1, prefix + "_points1.csv");
        savePointsToCSV(points2_noiseless_transformed, prefix + "_points2_noiseless_aligned.csv");
        savePointsToCSV(points2_noisy_transformed, prefix + "_points2_noisy_aligned.csv");

        cout << "Saved CSV files: " << prefix << "_*.csv\n";
        cout << "\n";
    }

    // Close results file
    results_file.close();
    cout << "Transformation matrices saved to scan_results.csv\n";
    cout << "=== All scan matching complete ===\n";

    return 0;
}
