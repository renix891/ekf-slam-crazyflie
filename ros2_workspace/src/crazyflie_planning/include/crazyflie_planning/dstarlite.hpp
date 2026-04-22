#ifndef CRAZYFLIE_PLANNING_DSTARLITE_HPP
#define CRAZYFLIE_PLANNING_DSTARLITE_HPP

#include <vector>
#include <queue>
#include <unordered_map>
#include <utility>
#include <limits>
#include <functional>
#include <mutex>

namespace crazyflie_planning {

/**
 * @brief Node structure for D* Lite algorithm
 */
struct DStarNode {
    int x;
    int y;
    double cost;

    DStarNode(int x_ = 0, int y_ = 0, double cost_ = 0.0)
        : x(x_), y(y_), cost(cost_) {}

    bool operator==(const DStarNode& other) const {
        return x == other.x && y == other.y;
    }
};

/**
 * @brief Hash function for DStarNode (for unordered_map)
 */
struct DStarNodeHash {
    std::size_t operator()(const DStarNode& node) const {
        return std::hash<int>()(node.x) ^ (std::hash<int>()(node.y) << 1);
    }
};

/**
 * @brief Priority key for D* Lite priority queue
 */
struct PriorityKey {
    double first;
    double second;

    PriorityKey(double f = 0.0, double s = 0.0) : first(f), second(s) {}

    bool operator<(const PriorityKey& other) const {
        return first < other.first || (first == other.first && second < other.second);
    }

    bool operator>(const PriorityKey& other) const {
        return first > other.first || (first == other.first && second > other.second);
    }
};

/**
 * @brief Priority queue element for D* Lite
 */
struct PriorityQueueElement {
    DStarNode node;
    PriorityKey key;

    bool operator>(const PriorityQueueElement& other) const {
        return key > other.key;
    }
};

/**
 * @brief D* Lite path planning algorithm for occupancy grids
 *
 * D* Lite is an incremental search algorithm that efficiently handles
 * dynamic environments by reusing information from previous searches.
 */
class DStarLitePlanner {
public:
    /**
     * @brief Constructor
     * @param width Grid width
     * @param height Grid height
     */
    DStarLitePlanner(int width, int height);

    /**
     * @brief Set cost map from occupancy grid
     * @param cost_map Cost map (width x height)
     */
    void setCostMap(const std::vector<std::vector<double>>& cost_map);

    /**
     * @brief Compute path from start to goal
     * @param start_x Start x coordinate (grid)
     * @param start_y Start y coordinate (grid)
     * @param goal_x Goal x coordinate (grid)
     * @param goal_y Goal y coordinate (grid)
     * @return Path as list of (x, y) grid coordinates
     */
    std::vector<std::pair<int, int>> computePath(int start_x, int start_y,
                                                  int goal_x, int goal_y);

    /**
     * @brief Get current path
     * @return Current path
     */
    const std::vector<std::pair<int, int>>& getCurrentPath() const {
        return current_path_;
    }

private:
    // Grid dimensions
    int x_max_;
    int y_max_;

    // D* Lite state
    DStarNode start_;
    DStarNode goal_;
    std::priority_queue<PriorityQueueElement,
                       std::vector<PriorityQueueElement>,
                       std::greater<PriorityQueueElement>> U_;
    double km_;

    // State grids (stored as flat arrays for efficiency)
    std::vector<double> rhs_;
    std::vector<double> g_;

    // Cost map
    std::vector<std::vector<double>> cost_map_;
    std::vector<std::vector<double>> old_cost_map_;
    bool initialized_;

    // Current path
    std::vector<std::pair<int, int>> current_path_;

    // Thread safety
    std::mutex planning_lock_;

    // Motion primitives (8-directional)
    static const std::vector<DStarNode> motions_;

    /**
     * @brief Initialize D* Lite search
     * @param start Start node
     * @param goal Goal node
     */
    void initialize(const DStarNode& start, const DStarNode& goal);

    /**
     * @brief Handle cost changes in the map
     * @param changed_cells List of changed cell coordinates
     */
    void handleCostChanges(const std::vector<std::pair<int, int>>& changed_cells);

    /**
     * @brief Cost of moving from node1 to node2
     * @param node1 Source node
     * @param node2 Target node
     * @return Cost (inf if blocked)
     */
    double c(const DStarNode& node1, const DStarNode& node2) const;

    /**
     * @brief Heuristic function (Chebyshev distance)
     * @param s Node
     * @return Heuristic distance to start
     */
    double h(const DStarNode& s) const;

    /**
     * @brief Calculate priority key for a node
     * @param s Node
     * @return Priority key
     */
    PriorityKey calculateKey(const DStarNode& s) const;

    /**
     * @brief Check if node is within grid bounds
     * @param node Node to check
     * @return True if valid
     */
    bool isValid(const DStarNode& node) const;

    /**
     * @brief Get neighbors of a node
     * @param u Node
     * @return List of neighbors
     */
    std::vector<DStarNode> getNeighbours(const DStarNode& u) const;

    /**
     * @brief Get predecessors of a node (same as neighbors in grid)
     * @param u Node
     * @return List of predecessors
     */
    std::vector<DStarNode> pred(const DStarNode& u) const;

    /**
     * @brief Get successors of a node (same as neighbors in grid)
     * @param u Node
     * @return List of successors
     */
    std::vector<DStarNode> succ(const DStarNode& u) const;

    /**
     * @brief Update vertex rhs value and priority queue
     * @param u Node to update
     */
    void updateVertex(const DStarNode& u);

    /**
     * @brief Compute shortest path using D* Lite
     */
    void computeShortestPath();

    /**
     * @brief Convert 2D coordinates to flat array index
     * @param x X coordinate
     * @param y Y coordinate
     * @return Flat array index
     */
    int toIndex(int x, int y) const {
        return y * x_max_ + x;
    }

    /**
     * @brief Get rhs value for a node
     * @param node Node
     * @return rhs value
     */
    double getRHS(const DStarNode& node) const {
        return rhs_[toIndex(node.x, node.y)];
    }

    /**
     * @brief Set rhs value for a node
     * @param node Node
     * @param value rhs value
     */
    void setRHS(const DStarNode& node, double value) {
        rhs_[toIndex(node.x, node.y)] = value;
    }

    /**
     * @brief Get g value for a node
     * @param node Node
     * @return g value
     */
    double getG(const DStarNode& node) const {
        return g_[toIndex(node.x, node.y)];
    }

    /**
     * @brief Set g value for a node
     * @param node Node
     * @param value g value
     */
    void setG(const DStarNode& node, double value) {
        g_[toIndex(node.x, node.y)] = value;
    }
};

}  // namespace crazyflie_planning

#endif  // CRAZYFLIE_PLANNING_DSTARLITE_HPP
