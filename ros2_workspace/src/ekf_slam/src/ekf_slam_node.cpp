#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "ekf_slam/ekf_core.hpp"

#include <cmath>
#include <vector>

class EKFSlamNode : public rclcpp::Node {
public:
    EKFSlamNode() : Node("ekf_slam_node"), last_odom_time_(0.0), prev_scan_pose_(Eigen::Vector3d::Zero()) {
        Eigen::Matrix3d process_noise   = Eigen::Matrix3d::Identity() * 0.01;
        // Bumped 0.05 -> 0.5: 4-beam translation-only scan match underestimates
        // displacement (centroid is pulled toward symmetry of the box room),
        // so trust odometry more and the scan match correction less.
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

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/ekf_pose", 10);
        pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/ekf_covariance", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&EKFSlamNode::publishPose, this));

        RCLCPP_INFO(this->get_logger(), "EKF-SLAM (scan-match) node initialized");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        double vx    = msg->twist.twist.linear.x;
        double vy    = msg->twist.twist.linear.y;
        double omega = msg->twist.twist.angular.z;

        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        if (last_odom_time_ > 0.0) {
            double dt = current_time - last_odom_time_;
            if (dt > 0.0 && dt < 1.0) {
                ekf_->predict(vx, vy, omega, dt);
            }
        }
        last_odom_time_ = current_time;
    }

    // Translation-only scan matching for sparse 4-beam multiranger data.
    //
    // Full SVD (rotation + translation) is poorly observable with only four
    // points roughly bounding-boxing the room — yaw and diagonal translation
    // alias.  We trust odometry for rotation and only solve translation:
    //     R = I,  t = centroid(points1) - centroid(points2)
    // Returns the same 3x3 homogeneous T contract as before, with the
    // top-left 2x2 fixed to identity.
    static Eigen::Matrix3d scanMatch(const Eigen::Matrix2Xd& points1,
                                     const Eigen::Matrix2Xd& points2) {
        Eigen::Vector2d c1 = points1.rowwise().mean();
        Eigen::Vector2d c2 = points2.rowwise().mean();

        Eigen::Vector2d t = c1 - c2;

        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        T.block<2, 1>(0, 2) = t;
        return T;
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // Crazyflie multiranger: 4 beams [back, right, front, left]
        if (msg->ranges.size() != 4) {
            RCLCPP_WARN(this->get_logger(),
                "Expected 4 range measurements, got %zu", msg->ranges.size());
            return;
        }

        const double bearings[4] = { M_PI, -M_PI/2, 0.0, M_PI/2 };

        // Convert each valid beam to a 2D point in the robot body frame.
        std::vector<Eigen::Vector2d> pts;
        pts.reserve(4);
        for (size_t i = 0; i < 4; ++i) {
            double r = msg->ranges[i];
            if (!std::isfinite(r) || r < msg->range_min || r > msg->range_max) continue;
            pts.emplace_back(r * std::cos(bearings[i]), r * std::sin(bearings[i]));
        }

        // Need at least 2 points for SVD scan matching to be meaningful.
        if (pts.size() < 2) {
            return;
        }

        Eigen::Matrix2Xd current(2, pts.size());
        for (size_t i = 0; i < pts.size(); ++i) current.col(i) = pts[i];

        // First scan: store and bail.
        if (!ekf_->hasPreviousScan_) {
            ekf_->previousScan_    = current;
            ekf_->hasPreviousScan_ = true;
            prev_scan_pose_        = ekf_->getPose();
            return;
        }

        // Scan matching needs corresponding columns. The 4-beam multiranger
        // gives ordered beams; if a beam dropped in/out between scans, sizes
        // can differ.  Skip the update in that case rather than mis-correspond.
        if (ekf_->previousScan_.cols() != current.cols()) {
            ekf_->previousScan_    = current;
            ekf_->hasPreviousScan_ = true;
            prev_scan_pose_        = ekf_->getPose();
            return;
        }

        // --- Rotation guard ---
        // With only 4 beams the SVD cannot disambiguate rotation from
        // translation; if the EKF (driven by odom predict) already says we've
        // yawed appreciably since the previous scan, refresh the reference
        // and skip the correction this iteration.
        Eigen::Vector3d cur_pose = ekf_->getPose();
        double yaw_diff = cur_pose(2) - prev_scan_pose_(2);
        while (yaw_diff >  M_PI) yaw_diff -= 2.0 * M_PI;
        while (yaw_diff < -M_PI) yaw_diff += 2.0 * M_PI;
        if (std::fabs(yaw_diff) > 0.1) {
            ekf_->previousScan_ = current;
            prev_scan_pose_     = cur_pose;
            return;
        }

        // --- Minimum-motion guard ---
        // If the drone is hovering, scan match noise injects pure error.
        double trans_dist = std::hypot(cur_pose(0) - prev_scan_pose_(0),
                                       cur_pose(1) - prev_scan_pose_(1));
        if (trans_dist < 0.02) {
            return;
        }

        // Run SVD scan match: previousScan_ ~= T_rel * current (both in body frame
        // at their respective times). T_rel maps the *current* body frame back into
        // the *previous* body frame.
        Eigen::Matrix3d T_rel = scanMatch(ekf_->previousScan_, current);

        double rel_dx     = T_rel(0, 2);
        double rel_dy     = T_rel(1, 2);
        double rel_dtheta = std::atan2(T_rel(1, 0), T_rel(0, 0));

        // Mean residual after alignment (in the previous-scan body frame).
        Eigen::Matrix2Xd current_h(2, current.cols());
        for (int i = 0; i < current.cols(); ++i) {
            Eigen::Vector3d p_h(current(0, i), current(1, i), 1.0);
            Eigen::Vector3d q   = T_rel * p_h;
            current_h.col(i) = q.head<2>();
        }
        double mean_residual = (ekf_->previousScan_ - current_h).colwise().norm().mean();
        double match_quality = 1.0 / (1.0 + mean_residual);

        // Compose the relative transform onto the pose at the time of the previous
        // scan to produce an absolute world-frame pose observation.
        double px = prev_scan_pose_(0);
        double py = prev_scan_pose_(1);
        double pt = prev_scan_pose_(2);
        double cpt = std::cos(pt), spt = std::sin(pt);

        double obs_x     = px + cpt * rel_dx - spt * rel_dy;
        double obs_y     = py + spt * rel_dx + cpt * rel_dy;
        double obs_theta = pt + rel_dtheta;
        while (obs_theta >  M_PI) obs_theta -= 2.0 * M_PI;
        while (obs_theta < -M_PI) obs_theta += 2.0 * M_PI;

        ekf_->updateScanMatch(obs_x, obs_y, obs_theta, match_quality);

        RCLCPP_DEBUG(this->get_logger(),
            "scanmatch rel=(%.3f, %.3f, %.3f) abs=(%.3f, %.3f, %.3f) q=%.3f",
            rel_dx, rel_dy, rel_dtheta, obs_x, obs_y, obs_theta, match_quality);

        ekf_->previousScan_ = current;
        prev_scan_pose_     = ekf_->getPose();
    }

    void publishPose() {
        Eigen::Vector3d pose     = ekf_->getPose();
        Eigen::Matrix3d pose_cov = ekf_->getPoseCovariance();

        auto stamp = this->now();

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp    = stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = pose(0);
        pose_msg.pose.position.y = pose(1);
        pose_msg.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, pose(2));
        pose_msg.pose.orientation = tf2::toMsg(q);
        pose_pub_->publish(pose_msg);

        geometry_msgs::msg::PoseWithCovarianceStamped pose_cov_msg;
        pose_cov_msg.header    = pose_msg.header;
        pose_cov_msg.pose.pose = pose_msg.pose;
        for (int i = 0; i < 36; i++) pose_cov_msg.pose.covariance[i] = 0.0;
        pose_cov_msg.pose.covariance[0]  = pose_cov(0, 0);
        pose_cov_msg.pose.covariance[1]  = pose_cov(0, 1);
        pose_cov_msg.pose.covariance[5]  = pose_cov(0, 2);
        pose_cov_msg.pose.covariance[6]  = pose_cov(1, 0);
        pose_cov_msg.pose.covariance[7]  = pose_cov(1, 1);
        pose_cov_msg.pose.covariance[11] = pose_cov(1, 2);
        pose_cov_msg.pose.covariance[30] = pose_cov(2, 0);
        pose_cov_msg.pose.covariance[31] = pose_cov(2, 1);
        pose_cov_msg.pose.covariance[35] = pose_cov(2, 2);
        pose_cov_pub_->publish(pose_cov_msg);
    }

    std::unique_ptr<ekf_slam::EKFCore> ekf_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr  scan_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    double          last_odom_time_;
    Eigen::Vector3d prev_scan_pose_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
