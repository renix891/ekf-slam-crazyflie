#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>

#include "ekf_slam/ekf_core.hpp"
#include "ekf_slam/line_extractor.hpp"

#include <cmath>
#include <limits>
#include <vector>

class EKFSlamNode : public rclcpp::Node {
public:
    EKFSlamNode() : Node("ekf_slam_node"), last_odom_time_(0.0) {
        Eigen::Matrix4d process_noise   = Eigen::Matrix4d::Identity() * 0.01;
        // Scan-match noise param is no longer used; kept to preserve EKFCore
        // ctor signature.
        Eigen::Matrix3d scanmatch_noise = Eigen::Matrix3d::Identity() * 0.5;

        ekf_ = std::make_unique<ekf_slam::EKFCore>(process_noise, scanmatch_noise);

        // Q_line: measurement noise for (rho, theta) line observations in robot frame.
        // Diagonal-constant per Stage-2 line extractor spec, with a 2x inflation
        // applied here so multi-bucket observations don't co-correct the pose
        // beyond what the underlying point uncertainty justifies.
        Q_line_ = Eigen::Matrix2d::Zero();
        Q_line_(0, 0) = 0.04;   // 2 * 0.02
        Q_line_(1, 1) = 0.02;   // 2 * 0.01

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

        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/ekf_pose", 10);
        pose_cov_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/ekf_covariance", 10);
        lines_dbg_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/lines", 10);
        corners_dbg_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/corners", 10);
        landmark_lines_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/landmark_lines", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&EKFSlamNode::publishPose, this));

        RCLCPP_INFO(this->get_logger(),
            "EKF-SLAM landmark mode initialized (line landmarks; max %d)",
            static_cast<int>(MAX_LINE_LANDMARKS));
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

    static constexpr size_t MAX_LINE_LANDMARKS = 20;
    static constexpr double CHI2_GATE_2DOF     = 5.991;  // 95% threshold

    void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        if (msg->ranges.size() != 4) {
            RCLCPP_WARN(this->get_logger(),
                "Expected 4 range measurements, got %zu", msg->ranges.size());
            return;
        }

        const double bearings[4] = { M_PI, -M_PI/2, 0.0, M_PI/2 };

        std::vector<double> ranges(4);
        for (size_t i = 0; i < 4; ++i) ranges[i] = static_cast<double>(msg->ranges[i]);

        // Feed extractor with the latest scan (in current EKF pose's frame).
        Eigen::Vector4d pose = ekf_->getPose();
        std::array<double, 4> r_arr = {ranges[0], ranges[1], ranges[2], ranges[3]};
        std::array<double, 4> b_arr = {bearings[0], bearings[1], bearings[2], bearings[3]};
        line_extractor_.addScan(pose(0), pose(1), pose(3), r_arr, b_arr);

        std::vector<ekf_slam::LineObs>   lines;
        std::vector<ekf_slam::CornerObs> corners;
        line_extractor_.extract(pose(0), pose(1), pose(3), lines, corners);

        // Raw extractor output (for RViz inspection of what the front-end sees).
        publishLineMarkers(lines);
        publishCornerMarkers(corners);

        // Run SLAM data association + augment/update for each detected line.
        runLineSlamUpdate(lines);

        // Persisted state landmarks (post-update) for RViz inspection.
        publishStateLandmarkLines();

        // Per-scan log: landmark count, state dim, last line.
        logSlamState(lines);
    }

    /// For each observed line: gate against existing landmarks; if the best
    /// Mahalanobis distance falls below CHI2_GATE_2DOF, run an EKF update on
    /// the matched landmark. Otherwise, augment a new landmark up to the cap.
    void runLineSlamUpdate(const std::vector<ekf_slam::LineObs>& lines) {
        for (const auto& obs : lines) {
            const double rho_obs   = obs.rho;
            const double theta_obs = obs.theta;

            const int n_lm   = ekf_->nLandmarks2();
            int   best_j     = -1;
            double best_d2   = std::numeric_limits<double>::infinity();

            // Mahalanobis test against every existing line landmark.
            for (int k = 0; k < n_lm; ++k) {
                int idx = 4 + 2 * k;
                Eigen::Vector2d z_pred;
                Eigen::MatrixXd H;
                ekf_->predictLineObservation(idx, z_pred, H);
                Eigen::Vector2d nu;
                nu(0) = rho_obs - z_pred(0);
                nu(1) = ekf_slam::EKFCore::normalizeAngle(theta_obs - z_pred(1));
                Eigen::Matrix2d S = H * ekf_->getCovariance() * H.transpose() + Q_line_;
                double d2 = nu.transpose() * S.inverse() * nu;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_j  = idx;
                }
            }

            if (best_j >= 0 && best_d2 < CHI2_GATE_2DOF) {
                ekf_->updateLineLandmark(best_j, rho_obs, theta_obs, Q_line_);
            } else if (n_lm < static_cast<int>(MAX_LINE_LANDMARKS)) {
                ekf_->augmentLineFromObservation(rho_obs, theta_obs, Q_line_);
            }
            // else: cap reached, no association — drop this observation.
        }
    }

    void publishLineMarkers(const std::vector<ekf_slam::LineObs>& lines) {
        visualization_msgs::msg::MarkerArray arr;
        // Always publish a DELETEALL first so stale segments disappear.
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = this->now();
        del.action          = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(del);

        int id = 0;
        for (const auto& lo : lines) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id    = "map";
            m.header.stamp       = this->now();
            m.ns                 = "ekf_slam_lines";
            m.id                 = id++;
            m.type               = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action             = visualization_msgs::msg::Marker::ADD;
            m.scale.x            = 0.02;
            m.color.r            = 0.1f;
            m.color.g            = 0.9f;
            m.color.b            = 0.2f;
            m.color.a            = 1.0f;
            m.pose.orientation.w = 1.0;

            // Draw a 2 m segment centred at the closest point on the line.
            double cx_w = lo.rho_w * std::cos(lo.theta_w);
            double cy_w = lo.rho_w * std::sin(lo.theta_w);
            double tx   = -std::sin(lo.theta_w);
            double ty   =  std::cos(lo.theta_w);
            geometry_msgs::msg::Point p1, p2;
            p1.x = cx_w - tx; p1.y = cy_w - ty; p1.z = 0.0;
            p2.x = cx_w + tx; p2.y = cy_w + ty; p2.z = 0.0;
            m.points.push_back(p1);
            m.points.push_back(p2);
            arr.markers.push_back(m);
        }
        lines_dbg_pub_->publish(arr);
    }

    void publishCornerMarkers(const std::vector<ekf_slam::CornerObs>& corners) {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = this->now();
        del.action          = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(del);

        int id = 0;
        for (const auto& co : corners) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id    = "map";
            m.header.stamp       = this->now();
            m.ns                 = "ekf_slam_corners";
            m.id                 = id++;
            m.type               = visualization_msgs::msg::Marker::SPHERE;
            m.action             = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x    = co.xw;
            m.pose.position.y    = co.yw;
            m.pose.position.z    = 0.0;
            m.pose.orientation.w = 1.0;
            m.scale.x            = 0.10;
            m.scale.y            = 0.10;
            m.scale.z            = 0.10;
            m.color.r            = 0.95f;
            m.color.g            = 0.4f;
            m.color.b            = 0.1f;
            m.color.a            = 1.0f;
            arr.markers.push_back(m);
        }
        corners_dbg_pub_->publish(arr);
    }

    /// Render every persisted line landmark in the EKF state as a 2 m segment
    /// in the world frame so RViz can show how the back-end's belief diverges
    /// from raw extractor output.
    void publishStateLandmarkLines() {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = this->now();
        del.action          = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(del);

        const Eigen::VectorXd& mu = ekf_->getState();
        int n_lm = ekf_->nLandmarks2();
        for (int k = 0; k < n_lm; ++k) {
            int idx = 4 + 2 * k;
            double rho_w   = mu(idx);
            double theta_w = mu(idx + 1);

            visualization_msgs::msg::Marker m;
            m.header.frame_id    = "map";
            m.header.stamp       = this->now();
            m.ns                 = "ekf_slam_landmark_lines";
            m.id                 = k;
            m.type               = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action             = visualization_msgs::msg::Marker::ADD;
            m.scale.x            = 0.03;
            m.color.r            = 0.2f;
            m.color.g            = 0.5f;
            m.color.b            = 1.0f;
            m.color.a            = 1.0f;
            m.pose.orientation.w = 1.0;
            double cx_w = rho_w * std::cos(theta_w);
            double cy_w = rho_w * std::sin(theta_w);
            double tx   = -std::sin(theta_w);
            double ty   =  std::cos(theta_w);
            geometry_msgs::msg::Point p1, p2;
            p1.x = cx_w - tx; p1.y = cy_w - ty; p1.z = 0.0;
            p2.x = cx_w + tx; p2.y = cy_w + ty; p2.z = 0.0;
            m.points.push_back(p1);
            m.points.push_back(p2);
            arr.markers.push_back(m);
        }
        landmark_lines_pub_->publish(arr);
    }

    void logSlamState(const std::vector<ekf_slam::LineObs>& lines) {
        int n_lm = ekf_->nLandmarks2();
        int dim  = ekf_->stateDim();
        if (lines.empty()) {
            RCLCPP_INFO(this->get_logger(),
                "Line landmarks: %d | State dim: %d | (no obs this scan)",
                n_lm, dim);
        } else {
            const auto& last = lines.back();
            RCLCPP_INFO(this->get_logger(),
                "Line landmarks: %d | State dim: %d | Last obs: rho=%.3f theta=%.3f",
                n_lm, dim, last.rho, last.theta);
        }
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
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lines_dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr corners_dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_lines_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    ekf_slam::LineExtractor line_extractor_;
    Eigen::Matrix2d         Q_line_;

    double last_odom_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
