#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;

// Structure to hold line parameters (normal form: n·x = n0)
struct Line {
    Vector2d normal;  // Unit normal vector
    double offset;    // Distance from origin
};

// Generate N noisy 2D points on a line defined by normal n and offset n0
// Line equation: n·x = n0
// Noise is added according to covariance matrix cov
MatrixXd generatePoints(const Vector2d& n, double n0, int N, const Matrix2d& cov) {
    MatrixXd points(2, N);

    // Set up random number generator
    random_device rd;
    mt19937 gen(rd());

    // Create a distribution for the parameter along the line
    uniform_real_distribution<> param_dist(-5.0, 5.0);

    // Normalize the normal vector
    Vector2d n_normalized = n.normalized();

    // Find a point on the line (closest to origin)
    Vector2d point_on_line = n0 * n_normalized;

    // Find a direction vector perpendicular to normal (tangent to line)
    Vector2d tangent(-n_normalized(1), n_normalized(0));

    // Compute Cholesky decomposition of covariance for noise generation
    LLT<Matrix2d> llt(cov);
    Matrix2d L = llt.matrixL();

    // Standard normal distribution for noise
    normal_distribution<> noise_dist(0.0, 1.0);

    for (int i = 0; i < N; ++i) {
        // Generate a parameter value along the line
        double t = param_dist(gen);

        // Point on the ideal line
        Vector2d ideal_point = point_on_line + t * tangent;

        // Generate correlated Gaussian noise
        Vector2d noise;
        noise << noise_dist(gen), noise_dist(gen);
        noise = L * noise;

        // Add noise to get final point
        points.col(i) = ideal_point + noise;
    }

    return points;
}

// Fit a line using Ordinary Least Squares (y = mx + b)
// Returns line in normal form
Line fitOLS(const MatrixXd& points) {
    int N = points.cols();

    // Extract x and y coordinates
    VectorXd x = points.row(0);
    VectorXd y = points.row(1);

    // Compute means
    double x_mean = x.mean();
    double y_mean = y.mean();

    // Center the data
    VectorXd x_centered = x.array() - x_mean;
    VectorXd y_centered = y.array() - y_mean;

    // Compute slope: m = sum((x-x_mean)(y-y_mean)) / sum((x-x_mean)^2)
    double numerator = x_centered.dot(y_centered);
    double denominator = x_centered.dot(x_centered);

    double m = numerator / denominator;
    double b = y_mean - m * x_mean;

    // Convert y = mx + b to normal form: n·x = n0
    // mx - y + b = 0  =>  normal = [m, -1]^T (then normalize)
    Vector2d normal(m, -1.0);
    normal.normalize();

    // Compute offset: n0 = n·p for any point p on the line
    // Use the mean point
    Vector2d mean_point(x_mean, y_mean);
    double offset = normal.dot(mean_point);

    Line result;
    result.normal = normal;
    result.offset = offset;

    return result;
}

// Fit a line using Total Least Squares (SVD-based)
// Returns line in normal form
Line fitTLS(const MatrixXd& points) {
    int N = points.cols();

    // Compute mean of points
    Vector2d mean = points.rowwise().mean();

    // Center the points
    MatrixXd centered = points.colwise() - mean;

    // Perform SVD on centered data matrix
    JacobiSVD<MatrixXd> svd(centered, ComputeFullU | ComputeFullV);

    // Normal = left singular vector of SMALLEST singular value
    Vector2d normal = svd.matrixU().col(1);  // col(1) not col(0)
    normal.normalize();
    double offset = normal.dot(mean);

    Line result;
    result.normal = normal;
    result.offset = offset;

    return result;
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

// Save line to CSV file (generate points along the line for plotting)
void saveLineToCSV(const Line& line, const string& filename, double x_min, double x_max) {
    ofstream file(filename);
    file << "x,y\n";

    // Generate points along the line
    int num_points = 100;
    for (int i = 0; i < num_points; ++i) {
        double x = x_min + (x_max - x_min) * i / (num_points - 1);

        // From n·[x,y]^T = n0, solve for y:
        // n_x * x + n_y * y = n0
        // y = (n0 - n_x * x) / n_y
        double y;
        if (abs(line.normal(1)) > 1e-10) {
            y = (line.offset - line.normal(0) * x) / line.normal(1);
        } else {
            // Vertical line case: x = n0 / n_x
            x = line.offset / line.normal(0);
            y = x_min + (x_max - x_min) * i / (num_points - 1);
        }

        file << x << "," << y << "\n";
    }
    file.close();
}

int main() {
    // Test configuration
    const int N = 100;
    const double n0 = 1.0;
    Matrix2d cov = Matrix2d::Identity() * 0.01;  // diag(0.01, 0.01)

    // Four test normals
    vector<Vector2d> test_normals = {
        Vector2d(1.0, 0.0),
        Vector2d(0.6, 0.8),
        Vector2d(-0.6, 0.8),
        Vector2d(0.0, 1.0)
    };

    cout << "=== Problem 2: Line Fitting Comparison ===\n\n";

    for (size_t i = 0; i < test_normals.size(); ++i) {
        Vector2d n = test_normals[i];

        cout << "Test Case " << (i + 1) << ": normal = ["
             << n(0) << ", " << n(1) << "]^T, offset = " << n0 << "\n";
        cout << "----------------------------------------\n";

        // Generate noisy points
        MatrixXd points = generatePoints(n, n0, N, cov);

        // Fit using OLS
        Line ols_line = fitOLS(points);
        cout << "OLS Fit:\n";
        cout << "  Normal: [" << ols_line.normal(0) << ", " << ols_line.normal(1) << "]^T\n";
        cout << "  Offset: " << ols_line.offset << "\n";

        // Fit using TLS
        Line tls_line = fitTLS(points);
        cout << "TLS Fit:\n";
        cout << "  Normal: [" << tls_line.normal(0) << ", " << tls_line.normal(1) << "]^T\n";
        cout << "  Offset: " << tls_line.offset << "\n";

        // Save data to CSV files
        string case_name = "case" + to_string(i + 1);
        savePointsToCSV(points, case_name + "_points.csv");
        saveLineToCSV(ols_line, case_name + "_ols.csv", -6.0, 6.0);
        saveLineToCSV(tls_line, case_name + "_tls.csv", -6.0, 6.0);

        cout << "\nFiles saved: " << case_name << "_points.csv, "
             << case_name << "_ols.csv, " << case_name << "_tls.csv\n";
        cout << "\n";
    }

    

    return 0;
}
