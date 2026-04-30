#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "crazyflie_planning/dstarlite.hpp"

#include <memory>
#include <vector>
#include <cmath>

class DStarLitePathPlanningNode : public rclcpp::Node {
public:
    DStarLitePathPlanningNode() : Node("dstarlite_path_planning_node") {
        // Parameters
        this->declare_parameter("planning_frequency", 2.0);
        this->declare_parameter("flight_height", 0.5);

        planning_freq_ = this->get_parameter("planning_frequency").as_double();
        flight_height_ = this->get_parameter("flight_height").as_double();

        // QoS for map (transient local for late joiners)
        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1))
                           .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
                           .durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

        // Subscribers
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", map_qos,
            std::bind(&DStarLitePathPlanningNode::mapCallback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/ekf_pose", 10,
            std::bind(&DStarLitePathPlanningNode::poseCallback, this, std::placeholders::_1));

        goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&DStarLitePathPlanningNode::goalCallback, this, std::placeholders::_1));

        // Publishers
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
        goal_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 10);

        // Timers
        planning_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / planning_freq_)),
            std::bind(&DStarLitePathPlanningNode::planningLoop, this));

        status_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&DStarLitePathPlanningNode::statusLoop, this));

        RCLCPP_INFO(this->get_logger(), "D* Lite Path Planning Node initialized");
        RCLCPP_INFO(this->get_logger(), "Waiting for occupancy grid on /crazyflie/map...");
    }

private:
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        bool was_none = (occupancy_grid_msg_ == nullptr);
        occupancy_grid_msg_ = msg;

        // Initialize planner if first map received
        if (was_none) {
            planner_ = std::make_unique<crazyflie_planning::DStarLitePlanner>(
                msg->info.width, msg->info.height);

            RCLCPP_DEBUG(this->get_logger(),
                "Received first occupancy grid: %ux%u, res=%.3fm",
                msg->info.width, msg->info.height, msg->info.resolution);
        }

        // Convert occupancy grid to cost map
        if (planner_) {
            auto cost_map = occupancyToCostMap(*msg);
            planner_->setCostMap(cost_map);
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        bool was_none = (current_pose_ == nullptr);
        current_pose_ = msg;

        if (was_none) {
            RCLCPP_DEBUG(this->get_logger(),
                "Received first pose: (%.2f, %.2f, %.2f)",
                msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        }
    }

    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        goal_pose_ = msg;
        RCLCPP_INFO(this->get_logger(),
            "New goal received: (%.2f, %.2f)",
            msg->pose.position.x, msg->pose.position.y);
        publishGoalMarker();
    }

    void planningLoop() {
        if (!planner_ || !occupancy_grid_msg_ || !current_pose_ || !goal_pose_) {
            return;
        }

        double start_x = current_pose_->pose.position.x;
        double start_y = current_pose_->pose.position.y;
        double goal_x = goal_pose_->pose.position.x;
        double goal_y = goal_pose_->pose.position.y;

        auto start_grid = worldToGrid(start_x, start_y);
        auto goal_grid = worldToGrid(goal_x, goal_y);

        // Clamp to grid bounds if out of range
        if (!start_grid) {
            auto clamped = clampToGridBounds(start_x, start_y);
            if (clamped) {
                start_grid = worldToGrid(clamped->first, clamped->second);
            }
            if (!start_grid) return;
        }

        if (!goal_grid) {
            auto clamped = clampToGridBounds(goal_x, goal_y);
            if (clamped) {
                goal_grid = worldToGrid(clamped->first, clamped->second);
            }
            if (!goal_grid) return;
        }

        // Compute path
        auto path = planner_->computePath(
            start_grid->first, start_grid->second,
            goal_grid->first, goal_grid->second);

        // Publish path
        if (!path.empty()) {
            publishPath(path);
        }
    }

    void statusLoop() {
        std::vector<std::string> parts;

        if (!current_pose_) {
            parts.push_back("NO_POSE");
        } else {
            parts.push_back("pose=(" +
                std::to_string(static_cast<int>(current_pose_->pose.position.x)) + "," +
                std::to_string(static_cast<int>(current_pose_->pose.position.y)) + ")");
        }

        if (!occupancy_grid_msg_) {
            parts.push_back("NO_MAP");
        } else {
            parts.push_back("map=" +
                std::to_string(occupancy_grid_msg_->info.width) + "x" +
                std::to_string(occupancy_grid_msg_->info.height));
        }

        if (!goal_pose_) {
            parts.push_back("NO_GOAL");
        } else {
            parts.push_back("goal=(" +
                std::to_string(static_cast<int>(goal_pose_->pose.position.x)) + "," +
                std::to_string(static_cast<int>(goal_pose_->pose.position.y)) + ")");
        }

        if (planner_ && !planner_->getCurrentPath().empty()) {
            parts.push_back("path=" +
                std::to_string(planner_->getCurrentPath().size()) + " waypoints");
        } else {
            parts.push_back("NO_PATH");
        }

        std::string status = "Status: ";
        for (size_t i = 0; i < parts.size(); i++) {
            status += parts[i];
            if (i < parts.size() - 1) status += " | ";
        }

        RCLCPP_DEBUG(this->get_logger(), "%s", status.c_str());
    }

    std::vector<std::vector<double>> occupancyToCostMap(
        const nav_msgs::msg::OccupancyGrid& grid_msg) {

        int width = grid_msg.info.width;
        int height = grid_msg.info.height;

        // Initialize cost map
        std::vector<std::vector<double>> cost_map(width, std::vector<double>(height, 1.0));

        // Convert occupancy values to costs
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int8_t val = grid_msg.data[idx];

                if (val < 0) {
                    // Unknown: mild penalty
                    cost_map[x][y] = 3.0;
                } else if (val == 0) {
                    // Free: low cost
                    cost_map[x][y] = 1.0;
                } else if (val <= 50) {
                    // Weighted avoidance zone near obstacles: gentle ramp
                    // (3.0-10.0) — keeps some clearance preference while
                    // leaving the planner room to route around obstacles.
                    cost_map[x][y] = 3.0 + (val / 50.0) * 7.0;
                } else {
                    // Obstacle: infinite cost
                    cost_map[x][y] = std::numeric_limits<double>::infinity();
                }
            }
        }

        return cost_map;
    }

    std::optional<std::pair<int, int>> worldToGrid(double x, double y) {
        if (!occupancy_grid_msg_) {
            return std::nullopt;
        }

        const auto& info = occupancy_grid_msg_->info;
        int gx = static_cast<int>((x - info.origin.position.x) / info.resolution);
        int gy = static_cast<int>((y - info.origin.position.y) / info.resolution);

        if (gx >= 0 && gx < static_cast<int>(info.width) &&
            gy >= 0 && gy < static_cast<int>(info.height)) {
            return std::make_pair(gx, gy);
        }

        return std::nullopt;
    }

    std::pair<double, double> gridToWorld(int gx, int gy) {
        const auto& info = occupancy_grid_msg_->info;
        double wx = gx * info.resolution + info.origin.position.x;
        double wy = gy * info.resolution + info.origin.position.y;
        return {wx, wy};
    }

    std::optional<std::pair<double, double>> clampToGridBounds(double x, double y) {
        if (!occupancy_grid_msg_) {
            return std::nullopt;
        }

        const auto& info = occupancy_grid_msg_->info;
        double min_x = info.origin.position.x;
        double max_x = info.origin.position.x + (info.width * info.resolution);
        double min_y = info.origin.position.y;
        double max_y = info.origin.position.y + (info.height * info.resolution);

        double cx = std::max(min_x, std::min(x, max_x - info.resolution));
        double cy = std::max(min_y, std::min(y, max_y - info.resolution));

        return std::make_pair(cx, cy);
    }

    void publishPath(const std::vector<std::pair<int, int>>& path) {
        if (path.empty() || !occupancy_grid_msg_) {
            return;
        }

        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = this->now();
        path_msg.header.frame_id = occupancy_grid_msg_->header.frame_id;

        // Skip leading waypoints within START_SKIP_RADIUS of the current pose so
        // the navigator doesn't auto-complete a waypoint sitting on the start
        // grid cell and burn its scan-timeout before any real motion occurs.
        constexpr double START_SKIP_RADIUS = 0.15;
        const double cur_x = current_pose_ ? current_pose_->pose.position.x : 0.0;
        const double cur_y = current_pose_ ? current_pose_->pose.position.y : 0.0;
        const bool have_pose = static_cast<bool>(current_pose_);

        size_t start_idx = 0;
        if (have_pose) {
            for (; start_idx + 1 < path.size(); ++start_idx) {
                auto [wx, wy] = gridToWorld(path[start_idx].first, path[start_idx].second);
                if (std::hypot(wx - cur_x, wy - cur_y) >= START_SKIP_RADIUS) {
                    break;
                }
            }
        }

        for (size_t i = start_idx; i < path.size(); ++i) {
            auto [wx, wy] = gridToWorld(path[i].first, path[i].second);

            geometry_msgs::msg::PoseStamped ps;
            ps.header = path_msg.header;
            ps.pose.position.x = wx;
            ps.pose.position.y = wy;
            ps.pose.position.z = flight_height_;
            ps.pose.orientation.w = 1.0;

            path_msg.poses.push_back(ps);
        }

        if (path_msg.poses.empty()) {
            return;
        }

        // Replace the grid-snapped final pose with the exact goal so the
        // navigator's tight final tolerance is measured against the real
        // target, not the centre of whatever cell the goal landed in
        // (off by up to half a resolution otherwise).
        if (goal_pose_) {
            auto & last = path_msg.poses.back();
            last.pose.position.x = goal_pose_->pose.position.x;
            last.pose.position.y = goal_pose_->pose.position.y;
        }

        path_pub_->publish(path_msg);
    }

    void publishGoalMarker() {
        if (!goal_pose_) {
            return;
        }

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = goal_pose_->header.frame_id.empty() ?
                                  "world" : goal_pose_->header.frame_id;
        marker.header.stamp = this->now();
        marker.ns = "goal";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position = goal_pose_->pose.position;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;

        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;

        marker.lifetime = rclcpp::Duration::from_seconds(0);  // Forever

        goal_marker_pub_->publish(marker);
    }

    // Members
    std::unique_ptr<crazyflie_planning::DStarLitePlanner> planner_;

    nav_msgs::msg::OccupancyGrid::SharedPtr occupancy_grid_msg_;
    geometry_msgs::msg::PoseStamped::SharedPtr current_pose_;
    geometry_msgs::msg::PoseStamped::SharedPtr goal_pose_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;

    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_marker_pub_;

    rclcpp::TimerBase::SharedPtr planning_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;

    double planning_freq_;
    double flight_height_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DStarLitePathPlanningNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
