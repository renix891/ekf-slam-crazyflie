#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

using namespace std;

// Structure to hold grid metadata
struct GridInfo {
    double x_min, x_max, y_min, y_max;
    double resolution;
    int rows, cols;
};

// Read point cloud from file (ignore z coordinate)
vector<pair<double, double>> readPointCloud(const string& filename) {
    vector<pair<double, double>> points;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return points;
    }

    string line;
    int line_num = 0;
    while (getline(file, line)) {
        line_num++;

        // Skip empty lines
        if (line.empty()) continue;

        stringstream ss(line);
        string token;
        vector<double> values;

        // Parse comma-separated values
        try {
            while (getline(ss, token, ',')) {
                values.push_back(stod(token));
            }

            // We need at least x,y (ignore z if present)
            if (values.size() >= 2) {
                points.push_back({values[0], values[1]});
            }
        } catch (const exception& e) {
            // Skip malformed lines
            cerr << "Warning: Skipping malformed line " << line_num << endl;
            continue;
        }
    }
    file.close();

    cout << "Loaded " << points.size() << " points from " << filename << endl;
    return points;
}

// Build 2D occupancy grid using thresholding/binning approach
pair<vector<vector<int>>, GridInfo> buildOccupancyGrid(
    const vector<pair<double, double>>& points,
    double resolution) {

    if (points.empty()) {
        cerr << "Error: No points to process" << endl;
        return {{}, {}};
    }

    // Find bounds using minmax_element
    auto x_minmax = minmax_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    auto y_minmax = minmax_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    double x_min = x_minmax.first->first;
    double x_max = x_minmax.second->first;
    double y_min = y_minmax.first->second;
    double y_max = y_minmax.second->second;

    // Add 0.5 meter buffer on all sides
    const double buffer = 0.5;
    x_min -= buffer;
    x_max += buffer;
    y_min -= buffer;
    y_max += buffer;

    // Compute grid dimensions
    int cols = static_cast<int>(ceil((x_max - x_min) / resolution));
    int rows = static_cast<int>(ceil((y_max - y_min) / resolution));

    cout << "Grid bounds: x=[" << x_min << ", " << x_max << "], y=["
         << y_min << ", " << y_max << "]" << endl;
    cout << "Grid dimensions: " << rows << " rows × " << cols << " cols" << endl;

    // Create 2D grid initialized to zeros (hit counter)
    vector<vector<int>> hit_count(rows, vector<int>(cols, 0));

    // Count hits in each cell
    for (const auto& point : points) {
        double x = point.first;
        double y = point.second;

        // Compute grid indices
        int col = static_cast<int>((x - x_min) / resolution);
        int row = static_cast<int>((y - y_min) / resolution);

        // Bounds checking
        if (row >= 0 && row < rows && col >= 0 && col < cols) {
            hit_count[row][col]++;
        }
    }

    // Binarize: any cell with hits > 0 becomes occupied (1)
    vector<vector<int>> grid(rows, vector<int>(cols, 0));
    int occupied_cells = 0;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (hit_count[r][c] > 0) {
                grid[r][c] = 1;
                occupied_cells++;
            }
        }
    }

    // Store grid metadata
    GridInfo info;
    info.x_min = x_min;
    info.x_max = x_max;
    info.y_min = y_min;
    info.y_max = y_max;
    info.resolution = resolution;
    info.rows = rows;
    info.cols = cols;

    int total_cells = rows * cols;
    double occupancy_pct = 100.0 * occupied_cells / total_cells;

    cout << "Total cells: " << total_cells << endl;
    cout << "Occupied cells: " << occupied_cells << endl;
    cout << "Occupancy: " << occupancy_pct << "%" << endl;

    return {grid, info};
}

// Save grid to CSV file
void saveGridToCSV(const vector<vector<int>>& grid, const string& filename) {
    ofstream file(filename);

    for (size_t r = 0; r < grid.size(); ++r) {
        for (size_t c = 0; c < grid[r].size(); ++c) {
            file << grid[r][c];
            if (c < grid[r].size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    cout << "Saved grid to: " << filename << endl;
}

// Save point cloud to CSV file
void savePointCloudToCSV(const vector<pair<double, double>>& points,
                         const string& filename) {
    ofstream file(filename);
    file << "x,y\n";

    for (const auto& point : points) {
        file << point.first << "," << point.second << "\n";
    }

    file.close();
    cout << "Saved point cloud to: " << filename << endl;
}

int main() {
    cout << "=== Problem 4: Occupancy Grid Mapping ===\n\n";

    // Default grid resolution (meters)
    const double resolution = 0.1;

    // Test files
    vector<string> test_files = {
        "../OccupancyGridTest1.txt",
        "../OccupancyGridTest2.txt"
    };

    for (size_t i = 0; i < test_files.size(); ++i) {
        int test_num = i + 1;
        string filename = test_files[i];

        cout << "Processing Test " << test_num << ": " << filename << endl;
        cout << string(60, '-') << "\n";

        // Read point cloud
        auto points = readPointCloud(filename);

        if (points.empty()) {
            cout << "Failed to load points from " << filename << "\n\n";
            continue;
        }

        // Build occupancy grid
        cout << "Building occupancy grid (resolution = " << resolution << " m)..." << endl;
        auto [grid, info] = buildOccupancyGrid(points, resolution);

        if (grid.empty()) {
            cout << "Failed to build grid\n\n";
            continue;
        }

        // Save outputs
        string prefix = "test" + to_string(test_num);
        savePointCloudToCSV(points, prefix + "_pointcloud.csv");
        saveGridToCSV(grid, prefix + "_grid.csv");

        cout << "\n";
    }

    cout << "=== All occupancy grids complete ===\n";

    return 0;
}
