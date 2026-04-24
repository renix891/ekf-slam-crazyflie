#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "ekf_slam/ekf_core.hpp"

#include <cmath>

class EKFSlamNode : public rclcpp::Node {
public:
    EKFSlamNode() : Node("ekf_slam_node"), last_odom_time_(0.0) {
        // Initialize EKF with default noise parameters
        Eigen::Matrix3d process_noise = Eigen::Matrix3d::Identity() * 0.01;
        Eigen::Matrix2d measurement_noise = Eigen::Matrix2d::Identity() * 0.05;
        // Chi-squared 95% for 2 DOF is 5.991; widened to 9.0 for this sensor's noise
        double mahalanobis_threshold = 9.0;

        ekf_ = std::make_unique<ekf_slam::EKFCore>(
            process_noise, measurement_noise, mahalanobis_threshold);

        // Subscribers
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/crazyflie/odom", 10,
            std::bind(&EKFSlamNode::odomCallback, this, std::placeholders::_1));

        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/crazyflie/scan", 10,
            std::bind(&EKFSlamNode::scanCallback, this, std::placeholders::_1));

        // Publishers
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/ekf_pose", 10);

        pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/ekf_covariance", 10);

        // Timer for publishing pose at 10Hz
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&EKFSlamNode::publishPose, this));

        RCLCPP_INFO(this->get_logger(), "EKF-SLAM node initialized");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Extract velocities from odometry message
        // Odometry twist is in the child_frame (body frame)
        double vx = msg->twist.twist.linear.x;
        double vy = msg->twist.twist.linear.y;
        double omega = msg->twist.twist.angular.z;

        RCLCPP_INFO(this->get_logger(),
            "odom rx: vx=%.3f vy=%.3f omega=%.3f", vx, vy, omega);

        // Compute time step
        double current_time = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

        if (last_odom_time_ > 0.0) {
            double dt = current_time - last_odom_time_;

            // Sanity check on dt
            if (dt > 0.0 && dt < 1.0) {  // Ignore jumps > 1 second
                // Prediction step
                ekf_->predict(vx, vy, omega, dt);
            }
        }

        last_odom_time_ = current_time;
    }

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // LaserScan contains 4 beams in order: [back, right, front, left]
        // Corresponding angles in body frame:
        //   back: � (180�)
        //   right: -�/2 (-90�)
        //   front: 0 (0�)
        //   left: �/2 (90�)

        if (msg->ranges.size() != 4) {
            RCLCPP_WARN(this->get_logger(),
                "Expected 4 range measurements, got %zu", msg->ranges.size());
            return;
        }

        // Define bearing angles for each beam (in body frame)
        const double bearings[4] = {
            M_PI,      // back
            -M_PI/2,   // right
            0.0,       // front
            M_PI/2     // left
        };

        const char* directions[4] = {"back", "right", "front", "left"};

        int measurements_processed = 0;

        for (size_t i = 0; i < 4; i++) {
            double range = msg->ranges[i];

            // Skip invalid measurements (inf, nan, or out of bounds)
            if (!std::isfinite(range) || range < msg->range_min || range > msg->range_max) {
                continue;
            }

            double rho = range;
            double theta_obs = bearings[i];

            // Data association: find matching landmark or create new one
            int landmark_id = ekf_->associateLandmark(rho, theta_obs);

            if (landmark_id >= 0) {
                // Landmark exists - update
                ekf_->update(rho, theta_obs, landmark_id);
                RCLCPP_DEBUG(this->get_logger(),
                    "Update landmark %d (%s): rho=%.3f, theta=%.3f",
                    landmark_id, directions[i], rho, theta_obs);
            } else {
                // New landmark - add to map
                landmark_id = ekf_->addLandmark(rho, theta_obs);
                RCLCPP_INFO(this->get_logger(),
                    "Added landmark %d (%s): rho=%.3f, theta=%.3f",
                    landmark_id, directions[i], rho, theta_obs);
            }

            measurements_processed++;
        }

        if (measurements_processed > 0) {
            RCLCPP_DEBUG(this->get_logger(),
                "Processed %d measurements, total landmarks: %d",
                measurements_processed, ekf_->getNumLandmarks());
        }
    }

    void publishPose() {
        // Get current pose estimate from EKF
        Eigen::Vector3d pose = ekf_->getPose();
        Eigen::Matrix3d pose_cov = ekf_->getPoseCovariance();

        // Create timestamp
        auto stamp = this->now();

        // Publish PoseStamped
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = stamp;
        pose_msg.header.frame_id = "map";

        pose_msg.pose.position.x = pose(0);
        pose_msg.pose.position.y = pose(1);
        pose_msg.pose.position.z = 0.0;  // 2D SLAM

        // Convert yaw to quaternion
        tf2::Quaternion q;
        q.setRPY(0, 0, pose(2));
        pose_msg.pose.orientation = tf2::toMsg(q);

        pose_pub_->publish(pose_msg);

        // Publish PoseWithCovarianceStamped for RViz uncertainty visualization
        geometry_msgs::msg::PoseWithCovarianceStamped pose_cov_msg;
        pose_cov_msg.header = pose_msg.header;
        pose_cov_msg.pose.pose = pose_msg.pose;

        // Fill 6x6 covariance matrix (x, y, z, roll, pitch, yaw)
        // We only have covariance for x, y, yaw (indices 0, 1, 5)
        for (int i = 0; i < 36; i++) {
            pose_cov_msg.pose.covariance[i] = 0.0;
        }

        // x-x, x-y, x-yaw
        pose_cov_msg.pose.covariance[0] = pose_cov(0, 0);   // x-x
        pose_cov_msg.pose.covariance[1] = pose_cov(0, 1);   // x-y
        pose_cov_msg.pose.covariance[5] = pose_cov(0, 2);   // x-yaw

        // y-x, y-y, y-yaw
        pose_cov_msg.pose.covariance[6] = pose_cov(1, 0);   // y-x
        pose_cov_msg.pose.covariance[7] = pose_cov(1, 1);   // y-y
        pose_cov_msg.pose.covariance[11] = pose_cov(1, 2);  // y-yaw

        // yaw-x, yaw-y, yaw-yaw
        pose_cov_msg.pose.covariance[30] = pose_cov(2, 0);  // yaw-x
        pose_cov_msg.pose.covariance[31] = pose_cov(2, 1);  // yaw-y
        pose_cov_msg.pose.covariance[35] = pose_cov(2, 2);  // yaw-yaw

        pose_cov_pub_->publish(pose_cov_msg);
    }

    // EKF instance
    std::unique_ptr<ekf_slam::EKFCore> ekf_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // State
    double last_odom_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
