#include "crazyflie_planning/dstarlite.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace crazyflie_planning {

// Define motion primitives (8-directional movement)
const std::vector<DStarNode> DStarLitePlanner::motions_ = {
    DStarNode(1, 0, 1.0),
    DStarNode(0, 1, 1.0),
    DStarNode(-1, 0, 1.0),
    DStarNode(0, -1, 1.0),
    DStarNode(1, 1, 1.414),  // sqrt(2) for diagonal
    DStarNode(1, -1, 1.414),
    DStarNode(-1, 1, 1.414),
    DStarNode(-1, -1, 1.414)
};

DStarLitePlanner::DStarLitePlanner(int width, int height)
    : x_max_(width),
      y_max_(height),
      km_(0.0),
      initialized_(false) {
    // Initialize grids
    int grid_size = x_max_ * y_max_;
    rhs_.resize(grid_size, std::numeric_limits<double>::infinity());
    g_.resize(grid_size, std::numeric_limits<double>::infinity());

    // Initialize cost map
    cost_map_.resize(x_max_, std::vector<double>(y_max_, 1.0));
    old_cost_map_.resize(x_max_, std::vector<double>(y_max_, 1.0));
}

void DStarLitePlanner::setCostMap(const std::vector<std::vector<double>>& cost_map) {
    std::lock_guard<std::mutex> lock(planning_lock_);

    old_cost_map_ = cost_map_;
    cost_map_ = cost_map;

    // Detect changes if initialized
    if (initialized_ && !old_cost_map_.empty()) {
        std::vector<std::pair<int, int>> changed_cells;

        for (int x = 0; x < x_max_; x++) {
            for (int y = 0; y < y_max_; y++) {
                if (old_cost_map_[x][y] != cost_map_[x][y]) {
                    changed_cells.push_back({x, y});
                }
            }
        }

        if (!changed_cells.empty()) {
            handleCostChanges(changed_cells);
        }
    }
}

void DStarLitePlanner::initialize(const DStarNode& start, const DStarNode& goal) {
    bool goal_changed = initialized_ && !(goal_ == goal);

    start_ = start;
    goal_ = goal;

    if (!initialized_ || goal_changed) {
        initialized_ = true;

        // Clear priority queue
        U_ = std::priority_queue<PriorityQueueElement,
                                 std::vector<PriorityQueueElement>,
                                 std::greater<PriorityQueueElement>>();

        km_ = 0.0;

        // Reset grids
        std::fill(rhs_.begin(), rhs_.end(), std::numeric_limits<double>::infinity());
        std::fill(g_.begin(), g_.end(), std::numeric_limits<double>::infinity());

        // Initialize goal
        setRHS(goal_, 0.0);

        // Add goal to priority queue
        PriorityQueueElement elem;
        elem.node = goal_;
        elem.key = calculateKey(goal_);
        U_.push(elem);
    }
}

void DStarLitePlanner::handleCostChanges(const std::vector<std::pair<int, int>>& changed_cells) {
    if (!initialized_) {
        return;
    }

    // Update km
    km_ += h(start_);

    // Update affected vertices
    for (const auto& cell : changed_cells) {
        int x = cell.first;
        int y = cell.second;

        if (x < 0 || x >= x_max_ || y < 0 || y >= y_max_) {
            continue;
        }

        DStarNode u(x, y);
        updateVertex(u);

        // Update neighbors
        for (const auto& neighbor : getNeighbours(u)) {
            updateVertex(neighbor);
        }
    }
}

double DStarLitePlanner::c(const DStarNode& node1, const DStarNode& node2) const {
    // Check if node2 is within bounds
    if (!isValid(node2)) {
        return std::numeric_limits<double>::infinity();
    }

    // Get cost from cost map
    double map_cost = cost_map_[node2.x][node2.y];

    // If cost is infinite (obstacle), can't traverse
    if (std::isinf(map_cost)) {
        return std::numeric_limits<double>::infinity();
    }

    // Calculate motion cost
    DStarNode motion(node1.x - node2.x, node1.y - node2.y);

    // Find matching motion primitive
    for (const auto& m : motions_) {
        if (m == motion) {
            // Total cost = motion cost * cell cost
            return m.cost * map_cost;
        }
    }

    return std::numeric_limits<double>::infinity();
}

double DStarLitePlanner::h(const DStarNode& s) const {
    // Chebyshev distance (max of dx, dy) - admissible for 8-connected grid
    return std::max(std::abs(start_.x - s.x), std::abs(start_.y - s.y));
}

PriorityKey DStarLitePlanner::calculateKey(const DStarNode& s) const {
    double min_g_rhs = std::min(getG(s), getRHS(s));
    return PriorityKey(min_g_rhs + h(s) + km_, min_g_rhs);
}

bool DStarLitePlanner::isValid(const DStarNode& node) const {
    return node.x >= 0 && node.x < x_max_ && node.y >= 0 && node.y < y_max_;
}

std::vector<DStarNode> DStarLitePlanner::getNeighbours(const DStarNode& u) const {
    std::vector<DStarNode> neighbors;

    for (const auto& motion : motions_) {
        DStarNode neighbor(u.x + motion.x, u.y + motion.y);
        if (isValid(neighbor)) {
            neighbors.push_back(neighbor);
        }
    }

    return neighbors;
}

std::vector<DStarNode> DStarLitePlanner::pred(const DStarNode& u) const {
    return getNeighbours(u);
}

std::vector<DStarNode> DStarLitePlanner::succ(const DStarNode& u) const {
    return getNeighbours(u);
}

void DStarLitePlanner::updateVertex(const DStarNode& u) {
    // Update rhs value (unless it's the goal)
    if (!(u == goal_)) {
        double min_cost = std::numeric_limits<double>::infinity();

        for (const auto& s_prime : succ(u)) {
            double cost = c(u, s_prime) + getG(s_prime);
            if (cost < min_cost) {
                min_cost = cost;
            }
        }

        setRHS(u, min_cost);
    }

    // Remove u from priority queue if present (inefficient but simple)
    // In production, use a more efficient data structure
    std::priority_queue<PriorityQueueElement,
                       std::vector<PriorityQueueElement>,
                       std::greater<PriorityQueueElement>> new_U;

    while (!U_.empty()) {
        auto elem = U_.top();
        U_.pop();
        if (!(elem.node == u)) {
            new_U.push(elem);
        }
    }
    U_ = new_U;

    // If inconsistent, add to priority queue
    if (getG(u) != getRHS(u)) {
        PriorityQueueElement elem;
        elem.node = u;
        elem.key = calculateKey(u);
        U_.push(elem);
    }
}

void DStarLitePlanner::computeShortestPath() {
    if (!initialized_) {
        return;
    }

    while (!U_.empty()) {
        PriorityKey k_start = calculateKey(start_);
        PriorityKey k_top = U_.top().key;

        // Check termination condition
        if (!(k_top < k_start) && getRHS(start_) == getG(start_)) {
            break;
        }

        PriorityQueueElement top_elem = U_.top();
        U_.pop();

        DStarNode u = top_elem.node;
        PriorityKey k_old = top_elem.key;
        PriorityKey k_new = calculateKey(u);

        if (k_old < k_new) {
            // Key needs updating
            PriorityQueueElement new_elem;
            new_elem.node = u;
            new_elem.key = k_new;
            U_.push(new_elem);
        } else if (getG(u) > getRHS(u)) {
            // Underconsistent vertex
            setG(u, getRHS(u));
            for (const auto& s : pred(u)) {
                updateVertex(s);
            }
        } else {
            // Overconsistent vertex
            setG(u, std::numeric_limits<double>::infinity());
            updateVertex(u);
            for (const auto& s : pred(u)) {
                updateVertex(s);
            }
        }
    }
}

std::vector<std::pair<int, int>> DStarLitePlanner::computePath(
    int start_x, int start_y, int goal_x, int goal_y) {

    std::lock_guard<std::mutex> lock(planning_lock_);

    DStarNode start_node(start_x, start_y);
    DStarNode goal_node(goal_x, goal_y);

    // Initialize if needed or if goal changed
    if (!initialized_ || !(goal_ == goal_node)) {
        initialize(start_node, goal_node);
    }

    // Update start position
    start_ = start_node;

    // Compute shortest path
    computeShortestPath();

    // Extract path
    if (std::isinf(getG(start_))) {
        // No path found
        return {};
    }

    std::vector<std::pair<int, int>> path;
    DStarNode current = start_;
    int max_steps = x_max_ * y_max_;  // Prevent infinite loops
    int steps = 0;

    while (!(current == goal_) && steps < max_steps) {
        path.push_back({current.x, current.y});

        // Find best successor
        std::vector<DStarNode> successors = succ(current);
        if (successors.empty()) {
            break;
        }

        // Find successor with minimum cost
        DStarNode best_succ = successors[0];
        double best_cost = c(current, successors[0]) + getG(successors[0]);

        for (size_t i = 1; i < successors.size(); i++) {
            double cost = c(current, successors[i]) + getG(successors[i]);
            if (cost < best_cost) {
                best_cost = cost;
                best_succ = successors[i];
            }
        }

        // Check if path is blocked
        if (std::isinf(best_cost)) {
            break;
        }

        current = best_succ;
        steps++;
    }

    if (current == goal_) {
        path.push_back({goal_.x, goal_.y});
    }

    current_path_ = path;
    return path;
}

}  // namespace crazyflie_planning
