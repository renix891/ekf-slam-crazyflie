#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>

#include "ekf_slam/ekf_core.hpp"

#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

class EKFSlamNode : public rclcpp::Node {
public:
    EKFSlamNode() : Node("ekf_slam_node"), last_odom_time_(0.0), prev_scan_pose_(Eigen::Vector4d::Zero()) {
        Eigen::Matrix4d process_noise   = Eigen::Matrix4d::Identity() * 0.01;
        Eigen::Matrix3d scanmatch_noise = Eigen::Matrix3d::Identity() * 0.5;

        ekf_ = std::make_unique<ekf_slam::EKFCore>(process_noise, scanmatch_noise);

        RCLCPP_INFO(this->get_logger(),
            "Q_scanmatch diag = (%.3f, %.3f, %.3f)",
            scanmatch_noise(0, 0), scanmatch_noise(1, 1), scanmatch_noise(2, 2));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/crazyflie/odom", 10,
            std::bind(&EKFSlamNode::odomCallback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/crazyflie/scan", 10,
            std::bind(&EKFSlamNode::scanCallback, this, std::placeholders::_1));

        // Downward 1-beam TOF.  Independent of /crazyflie/odom — that's what
        // makes the z-update real fusion rather than tautology.
        down_range_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/crazyflie/range/down", 10,
            std::bind(&EKFSlamNode::downRangeCallback, this, std::placeholders::_1));

        // Tap the commanded velocity so updateZ can tell hover apart from
        // takeoff/landing and pick the right outlier threshold.
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&EKFSlamNode::cmdVelCallback, this, std::placeholders::_1));

        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", rclcpp::QoS(1).transient_local(),
            std::bind(&EKFSlamNode::mapCallback, this, std::placeholders::_1));

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/ekf_pose", 10);
        pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/ekf_covariance", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&EKFSlamNode::publishPose, this));

        RCLCPP_INFO(this->get_logger(),
            "EKF-SLAM (4-D state: x, y, z, theta) initialized");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double vx    = msg->twist.twist.linear.x;
        double vy    = msg->twist.twist.linear.y;
        double vz    = msg->twist.twist.linear.z;
        double omega = msg->twist.twist.angular.z;

        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        if (last_odom_time_ > 0.0) {
            double dt = current_time - last_odom_time_;
            if (dt > 0.0 && dt < 1.0) {
                ekf_->predict(vx, vy, vz, omega, dt);
            }
        }
        last_odom_time_ = current_time;

        // Use the odom-reported orientation as a direct yaw measurement.
        const auto& q = msg->pose.pose.orientation;
        tf2::Quaternion tfq(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw_from_odom;
        tf2::Matrix3x3(tfq).getRPY(roll, pitch, yaw_from_odom);
        ekf_->updateYaw(yaw_from_odom);
    }

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        ekf_->setCommandedVz(msg->linear.z);
    }

    void downRangeCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (msg->ranges.empty()) return;
        double r = static_cast<double>(msg->ranges[0]);
        if (!std::isfinite(r)) return;
        if (r < msg->range_min || r > msg->range_max) return;

        // Reject readings outside the plausible flight envelope. Below 0.05 m
        // is sensor minimum / propeller wash; above 2.0 m is out of indoor
        // operating range and likely a max-range fault.
        if (r < 0.05 || r > 2.0) return;

        // If the beam is significantly shorter than our current z estimate,
        // it almost certainly hit an obstacle on the floor, not the floor.
        // Skip the update so the EKF doesn't dive toward a phantom ground.
        double z_est = ekf_->getPose()(2);
        if (r < z_est - 0.1) return;

        // Body-frame -Z range. At small tilts (typical CF roll/pitch < 5 deg)
        // the cos(tilt) correction is < 0.4 % and we can use range as a direct
        // world-frame z observation. Use a larger noise (0.05) so the range
        // sensor is one of several voices, not the dominant one.
        ekf_->updateZ(r, 0.05);
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        current_map_ = msg;
        map_received_ = true;
    }

    struct ScanMatchResult {
        double dx;            // absolute world-frame x observation
        double dy;            // absolute world-frame y observation
        double dtheta;        // absolute world-frame theta (from odom yaw)
        double match_quality;
        bool   valid;
    };

    // Scan-to-map match: for each beam endpoint, find nearest occupied cell in
    // the map and accumulate the offset.  Rotation is taken from odometry yaw
    // (4-beam multiranger doesn't give us enough constraints for rotational
    // scan matching).
    ScanMatchResult scanToMapMatch(const std::vector<double>& ranges,
                                   const double bearings[4],
                                   double /*dtheta_odom*/)
    {
        ScanMatchResult result{0.0, 0.0, 0.0, 0.0, false};

        std::lock_guard<std::mutex> lock(map_mutex_);
        if (!map_received_ || !current_map_) return result;

        int occupied_count = 0;
        for (auto& cell : current_map_->data) {
            if (cell == 100) occupied_count++;
        }
        if (occupied_count < 50) return result;

        const double res = current_map_->info.resolution;
        const double ox  = current_map_->info.origin.position.x;
        const double oy  = current_map_->info.origin.position.y;
        const int width  = static_cast<int>(current_map_->info.width);
        const int height = static_cast<int>(current_map_->info.height);

        Eigen::Vector4d pose = ekf_->getPose();
        const double robot_x     = pose(0);
        const double robot_y     = pose(1);
        const double robot_theta = pose(3);

        const int search_cells = static_cast<int>(0.5 / res);  // 0.5 m radius
        const double accept_dist = 0.3;                        // m

        double sum_dx = 0.0, sum_dy = 0.0;
        double total_residual = 0.0;
        int valid_beams = 0;

        for (int i = 0; i < 4; ++i) {
            double range = ranges[i];
            if (!std::isfinite(range) || range < 0.01 || range > 3.49) continue;

            double beam_angle_world = robot_theta + bearings[i];
            double bx = robot_x + range * std::cos(beam_angle_world);
            double by = robot_y + range * std::sin(beam_angle_world);

            int cx = static_cast<int>((bx - ox) / res);
            int cy = static_cast<int>((by - oy) / res);

            double best_dist = std::numeric_limits<double>::infinity();
            double best_wx = bx, best_wy = by;

            for (int dy = -search_cells; dy <= search_cells; ++dy) {
                for (int dx_cell = -search_cells; dx_cell <= search_cells; ++dx_cell) {
                    int gx = cx + dx_cell;
                    int gy = cy + dy;
                    if (gx < 0 || gx >= width || gy < 0 || gy >= height) continue;
                    int idx = gy * width + gx;
                    if (current_map_->data[idx] == 100) {
                        double dist = std::sqrt(static_cast<double>(dx_cell * dx_cell + dy * dy)) * res;
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_wx = ox + (gx + 0.5) * res;
                            best_wy = oy + (gy + 0.5) * res;
                        }
                    }
                }
            }

            if (best_dist < accept_dist) {
                sum_dx += (best_wx - bx);
                sum_dy += (best_wy - by);
                total_residual += best_dist;
                valid_beams++;
            }
        }

        if (valid_beams < 2) return result;

        result.dx = robot_x + sum_dx / valid_beams;
        result.dy = robot_y + sum_dy / valid_beams;
        result.dtheta = robot_theta;
        result.match_quality = 1.0 / (1.0 + total_residual / valid_beams);
        result.valid = true;
        return result;
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (msg->ranges.size() != 4) {
            RCLCPP_WARN(this->get_logger(),
                "Expected 4 range measurements, got %zu", msg->ranges.size());
            return;
        }

        const double bearings[4] = { M_PI, -M_PI/2, 0.0, M_PI/2 };

        std::vector<double> ranges(4);
        for (size_t i = 0; i < 4; ++i) ranges[i] = static_cast<double>(msg->ranges[i]);

        Eigen::Vector4d cur_pose = ekf_->getPose();

        double trans_dist = std::hypot(cur_pose(0) - prev_scan_pose_(0),
                                       cur_pose(1) - prev_scan_pose_(1));
        if (prev_scan_pose_set_ && trans_dist < 0.02) {
            return;
        }

        double dtheta_odom = cur_pose(3) - prev_scan_pose_(3);
        while (dtheta_odom >  M_PI) dtheta_odom -= 2.0 * M_PI;
        while (dtheta_odom < -M_PI) dtheta_odom += 2.0 * M_PI;

        ScanMatchResult sm = scanToMapMatch(ranges, bearings, dtheta_odom);
        if (!sm.valid) {
            prev_scan_pose_ = cur_pose;
            prev_scan_pose_set_ = true;
            return;
        }

        ekf_->updateScanMatch(sm.dx, sm.dy, sm.dtheta, sm.match_quality);

        RCLCPP_DEBUG(this->get_logger(),
            "scan-to-map abs=(%.3f, %.3f, %.3f) q=%.3f",
            sm.dx, sm.dy, sm.dtheta, sm.match_quality);

        prev_scan_pose_ = ekf_->getPose();
        prev_scan_pose_set_ = true;
    }

    void publishPose() {
        Eigen::Vector4d pose     = ekf_->getPose();
        Eigen::Matrix4d pose_cov = ekf_->getPoseCovariance();

        auto stamp = this->now();

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp    = stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = pose(0);
        pose_msg.pose.position.y = pose(1);
        pose_msg.pose.position.z = pose(2);

        tf2::Quaternion q;
        q.setRPY(0, 0, pose(3));
        pose_msg.pose.orientation = tf2::toMsg(q);
        pose_pub_->publish(pose_msg);

        // PoseWithCovariance: 6x6 row-major over [x, y, z, roll, pitch, yaw].
        // EKF state is [x, y, z, theta], so theta maps to yaw (index 5).
        geometry_msgs::msg::PoseWithCovarianceStamped pose_cov_msg;
        pose_cov_msg.header    = pose_msg.header;
        pose_cov_msg.pose.pose = pose_msg.pose;
        for (int i = 0; i < 36; i++) pose_cov_msg.pose.covariance[i] = 0.0;
        // x row
        pose_cov_msg.pose.covariance[0]  = pose_cov(0, 0);
        pose_cov_msg.pose.covariance[1]  = pose_cov(0, 1);
        pose_cov_msg.pose.covariance[2]  = pose_cov(0, 2);
        pose_cov_msg.pose.covariance[5]  = pose_cov(0, 3);
        // y row
        pose_cov_msg.pose.covariance[6]  = pose_cov(1, 0);
        pose_cov_msg.pose.covariance[7]  = pose_cov(1, 1);
        pose_cov_msg.pose.covariance[8]  = pose_cov(1, 2);
        pose_cov_msg.pose.covariance[11] = pose_cov(1, 3);
        // z row
        pose_cov_msg.pose.covariance[12] = pose_cov(2, 0);
        pose_cov_msg.pose.covariance[13] = pose_cov(2, 1);
        pose_cov_msg.pose.covariance[14] = pose_cov(2, 2);
        pose_cov_msg.pose.covariance[17] = pose_cov(2, 3);
        // yaw row
        pose_cov_msg.pose.covariance[30] = pose_cov(3, 0);
        pose_cov_msg.pose.covariance[31] = pose_cov(3, 1);
        pose_cov_msg.pose.covariance[32] = pose_cov(3, 2);
        pose_cov_msg.pose.covariance[35] = pose_cov(3, 3);
        pose_cov_pub_->publish(pose_cov_msg);
    }

    std::unique_ptr<ekf_slam::EKFCore> ekf_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr    scan_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr    down_range_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr      cmd_vel_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr   map_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    double          last_odom_time_;
    Eigen::Vector4d prev_scan_pose_;
    bool            prev_scan_pose_set_ = false;

    nav_msgs::msg::OccupancyGrid::SharedPtr current_map_;
    bool                                    map_received_ = false;
    std::mutex                              map_mutex_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
