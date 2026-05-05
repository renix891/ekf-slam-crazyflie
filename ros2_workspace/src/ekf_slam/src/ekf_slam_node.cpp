#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
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
#include <mutex>
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

        // Q_corner: 2x the extractor's per-axis variance (0.04) for the same
        // reason — front-end noise is correlated across the simultaneous line
        // and corner observations from a single scan, so corner updates get
        // an inflated noise to avoid double-counting.
        Q_corner_ = Eigen::Matrix2d::Zero();
        Q_corner_(0, 0) = 0.08;
        Q_corner_(1, 1) = 0.08;

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
        lines_dbg_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/lines", 10);
        corners_dbg_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/corners", 10);
        landmark_lines_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/landmark_lines", 10);
        landmark_corners_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/ekf_slam/debug/landmark_corners", 10);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&EKFSlamNode::publishPose, this));

        RCLCPP_INFO(this->get_logger(),
            "EKF-SLAM hybrid initialized: scan-to-map primary + line landmarks "
            "(max %d) + corner landmarks (max %d)",
            static_cast<int>(MAX_LINE_LANDMARKS),
            static_cast<int>(MAX_CORNER_LANDMARKS));
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

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        current_map_  = msg;
        map_received_ = true;
    }

    struct ScanMatchResult {
        double dx;            // absolute world-frame x observation
        double dy;            // absolute world-frame y observation
        double dtheta;        // absolute world-frame theta (passthrough from yaw)
        double match_quality;
        bool   valid;
    };

    // Scan-to-map nearest-occupied-cell ICP (4-beam variant). Returns an
    // absolute (x, y, theta) pose observation that updateScanMatch consumes.
    ScanMatchResult scanToMapMatch(const std::vector<double>& ranges,
                                   const double bearings[4]) {
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

        const int    search_cells = static_cast<int>(0.5 / res);  // 0.5 m radius
        const double accept_dist  = 0.3;                           // m

        double sum_dx = 0.0, sum_dy = 0.0;
        double total_residual = 0.0;
        int    valid_beams    = 0;

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
                            best_wx   = ox + (gx + 0.5) * res;
                            best_wy   = oy + (gy + 0.5) * res;
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

        result.dx            = robot_x + sum_dx / valid_beams;
        result.dy            = robot_y + sum_dy / valid_beams;
        result.dtheta        = robot_theta;  // passthrough; yaw is corrected via updateYaw
        result.match_quality = 1.0 / (1.0 + total_residual / valid_beams);
        result.valid         = true;
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

        // Stage 2 design: feed the extractor every scan so its world-bearing
        // buckets fill in regardless of whether the scan-to-map throttle gate
        // accepts this sample.
        Eigen::Vector4d pre_pose = ekf_->getPose();
        std::array<double, 4> r_arr = {ranges[0], ranges[1], ranges[2], ranges[3]};
        std::array<double, 4> b_arr = {bearings[0], bearings[1], bearings[2], bearings[3]};
        line_extractor_.addScan(pre_pose(0), pre_pose(1), pre_pose(3), r_arr, b_arr);

        // PRIMARY CORRECTION: scan-to-map. Same translation gate as before
        // Stage 3 — throttles the expensive nearest-occupied-cell search.
        Eigen::Vector4d cur_pose = ekf_->getPose();
        double trans_dist = std::hypot(cur_pose(0) - prev_scan_pose_(0),
                                       cur_pose(1) - prev_scan_pose_(1));
        bool ran_scan_to_map = false;
        bool gate_passed     = false;
        bool sm_valid        = false;
        double sm_quality    = 0.0;
        if (!prev_scan_pose_set_ || trans_dist >= 0.02) {
            gate_passed = true;
            ScanMatchResult sm = scanToMapMatch(ranges, bearings);
            sm_valid    = sm.valid;
            sm_quality  = sm.match_quality;
            if (sm.valid) {
                ekf_->updateScanMatch(sm.dx, sm.dy, sm.dtheta, sm.match_quality);
                ran_scan_to_map = true;
                if (!scan_to_map_initialized_) {
                    scan_to_map_initialized_ = true;
                    RCLCPP_INFO(this->get_logger(),
                        "[init] scan-to-map first success — line landmark "
                        "augment/update enabled from this scan onward.");
                }
                RCLCPP_DEBUG(this->get_logger(),
                    "scan-to-map abs=(%.3f, %.3f, %.3f) q=%.3f",
                    sm.dx, sm.dy, sm.dtheta, sm.match_quality);
            }
            prev_scan_pose_     = ekf_->getPose();
            prev_scan_pose_set_ = true;
        }

        // First-N diagnostic so we can tell whether scan2map=skipped means
        // (a) translation gate held, (b) gate passed but no map / not enough
        // occupied cells / not enough valid beams, or (c) map sub never fired.
        if (scans_seen_ < 5) {
            int occupied = 0;
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                if (current_map_) {
                    for (auto& cell : current_map_->data) if (cell == 100) occupied++;
                }
            }
            RCLCPP_INFO(this->get_logger(),
                "[scanCB diag #%lu] map_received=%d occupied_cells=%d trans_dist=%.4fm "
                "gate_passed=%d sm_valid=%d sm_q=%.2f ran_sm=%d",
                scans_seen_,
                static_cast<int>(map_received_),
                occupied,
                trans_dist,
                static_cast<int>(gate_passed),
                static_cast<int>(sm_valid),
                sm_quality,
                static_cast<int>(ran_scan_to_map));
        }
        scans_seen_++;

        // ADDITIONAL CORRECTION: line landmarks (true joint EKF-SLAM —
        // landmark update K is full-state-dim and correlates pose with all
        // landmarks). Q_line is diag(0.04, 0.02), 2x the extractor's
        // diagonal estimate, so scan-to-map and line updates don't
        // double-count the same beam endpoints.
        Eigen::Vector4d post_sm_pose = ekf_->getPose();
        std::vector<ekf_slam::LineObs>   lines;
        std::vector<ekf_slam::CornerObs> corners;
        line_extractor_.extract(post_sm_pose(0), post_sm_pose(1), post_sm_pose(3),
                                lines, corners);

        publishLineMarkers(lines);
        publishCornerMarkers(corners);

        runLineSlamUpdate(lines);
        runCornerSlamUpdate(corners);

        publishStateLandmarkLines();
        publishStateLandmarkCorners();
        logSlamState(lines, ran_scan_to_map);
    }

    // Confirmation buffer for new-landmark spawning. An observation is only
    // augmented to the EKF state once we have seen N_CONFIRM matches within
    // the confirm-gate of each other across distinct scans. Defends against
    // transient front-end fits running away with the state.
    static constexpr int    N_CONFIRM        = 3;
    static constexpr double CONFIRM_RHO_TOL  = 0.10;  // m
    static constexpr double CONFIRM_THETA_TOL = 0.10;  // rad
    static constexpr int    CANDIDATE_TTL    = 30;    // scans without re-obs before forget

    // Robot-frame rho bounds. Below RHO_MIN are near-origin garbage fits;
    // above RHO_MAX are outside the room and almost certainly extractor
    // noise (e.g. from a single bucket with two beam endpoints far apart).
    static constexpr double RHO_MIN_M = 0.15;
    static constexpr double RHO_MAX_M = 2.5;

    struct LineCandidate {
        double rho;
        double theta;
        int    hits;
        int    last_seen_scan;
    };
    std::vector<LineCandidate> line_candidates_;

    /// For each observed line: Mahalanobis-gate against existing landmarks;
    /// update if matched. If unmatched, route through the confirmation buffer
    /// and only augment once we have N_CONFIRM consistent observations.
    void runLineSlamUpdate(const std::vector<ekf_slam::LineObs>& lines) {
        // Hard gate: do not touch the landmark state until scan-to-map has
        // anchored the pose at least once. Otherwise the first scans run
        // against an uninitialized pose and seed the EKF with garbage.
        if (!scan_to_map_initialized_) return;

        // Per-scan diagnostic throttle (~1 Hz).
        bool diag_this_scan = false;
        double now_s = this->now().seconds();
        if (now_s - last_diag_s_ >= 1.0) {
            last_diag_s_   = now_s;
            diag_this_scan = true;
        }

        for (size_t obs_i = 0; obs_i < lines.size(); ++obs_i) {
            const auto& obs        = lines[obs_i];
            const double rho_obs   = obs.rho;
            const double theta_obs = obs.theta;

            // Robot-frame rho bounds: drop near-origin and out-of-room fits.
            if (std::abs(rho_obs) < RHO_MIN_M ||
                std::abs(rho_obs) > RHO_MAX_M) {
                if (diag_this_scan && obs_i == 0) {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA] obs=(rho=%.3f th=%.3f) REJECTED by rho gate "
                        "[%.2f, %.2f]m",
                        rho_obs, theta_obs, RHO_MIN_M, RHO_MAX_M);
                }
                continue;
            }

            const int n_lm    = ekf_->nLandmarks2();
            const int n_lines = ekf_->nLines();
            int   best_j     = -1;
            double best_d2   = std::numeric_limits<double>::infinity();
            Eigen::Vector2d best_z_pred(0, 0);
            Eigen::Vector2d best_nu(0, 0);

            // DA: predict only against line landmarks. Skip corner blocks.
            for (int k = 0; k < n_lm; ++k) {
                if (ekf_->landmarkKind(k) != ekf_slam::LandmarkKind::Line) continue;
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
                    best_d2     = d2;
                    best_j      = idx;
                    best_z_pred = z_pred;
                    best_nu     = nu;
                }
            }

            // Diagnostic: print first observation each scan so we can see why
            // association is failing. Includes obs, predicted, innovation, d2.
            if (diag_this_scan && obs_i == 0) {
                if (best_j >= 0) {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA] obs=(rho=%.3f th=%.3f) | nearest_lm idx=%d "
                        "z_pred=(%.3f, %.3f) innov=(%.3f, %.3f) d2=%.2f "
                        "(gate=%.2f) | n_lm=%d candidates=%zu",
                        rho_obs, theta_obs, best_j,
                        best_z_pred(0), best_z_pred(1),
                        best_nu(0), best_nu(1), best_d2,
                        CHI2_GATE_2DOF, n_lm, line_candidates_.size());
                } else {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA] obs=(rho=%.3f th=%.3f) | no existing landmarks | "
                        "candidates=%zu",
                        rho_obs, theta_obs, line_candidates_.size());
                }
            }

            if (best_j >= 0 && best_d2 < CHI2_GATE_2DOF) {
                bool applied = ekf_->updateLineLandmark(best_j, rho_obs, theta_obs, Q_line_);
                if (!applied) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                        1000,
                        "Line update skipped: S near-singular for landmark idx=%d "
                        "(landmark covariance has shrunk too far). Filter held "
                        "stable; will recover once Sigma grows.", best_j);
                } else if (!logged_first_update_) {
                    int N = ekf_->stateDim();
                    RCLCPP_INFO(this->get_logger(),
                        "[verify] first line landmark update applied. "
                        "state dim = %d, Kalman gain K dim = %dx2 (full-state). "
                        "True joint EKF-SLAM: pose and landmarks correlated.",
                        N, N);
                    logged_first_update_ = true;
                }
            } else if (n_lines < static_cast<int>(MAX_LINE_LANDMARKS)) {
                // Route through the confirmation buffer instead of augmenting
                // straight into the state.
                tryConfirmLineAndAugment(rho_obs, theta_obs);
            }
            // else: line cap reached, no association — drop this observation.
        }

        // Age out stale candidates.
        for (auto it = line_candidates_.begin(); it != line_candidates_.end();) {
            if (scans_seen_ - it->last_seen_scan > CANDIDATE_TTL) {
                it = line_candidates_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void tryConfirmLineAndAugment(double rho_obs, double theta_obs) {
        // Find the nearest candidate within the confirm tolerance.
        int best = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < line_candidates_.size(); ++i) {
            double drho = std::abs(rho_obs - line_candidates_[i].rho);
            double dth  = std::abs(ekf_slam::EKFCore::normalizeAngle(
                              theta_obs - line_candidates_[i].theta));
            if (drho < CONFIRM_RHO_TOL && dth < CONFIRM_THETA_TOL) {
                double d = drho + dth;
                if (d < best_dist) {
                    best_dist = d;
                    best      = static_cast<int>(i);
                }
            }
        }
        if (best < 0) {
            // New candidate.
            line_candidates_.push_back({rho_obs, theta_obs, 1,
                                        static_cast<int>(scans_seen_)});
            return;
        }
        // Existing candidate — refine running mean and bump hit count.
        auto& c = line_candidates_[best];
        // Don't double-count multiple obs in the same scan.
        if (c.last_seen_scan == static_cast<int>(scans_seen_)) return;
        c.hits           += 1;
        c.last_seen_scan  = static_cast<int>(scans_seen_);
        c.rho             = 0.5 * (c.rho + rho_obs);
        c.theta           = 0.5 * (c.theta + ekf_slam::EKFCore::normalizeAngle(
                                                  theta_obs));
        if (c.hits >= N_CONFIRM) {
            ekf_->augmentLineFromObservation(c.rho, c.theta, Q_line_);
            RCLCPP_INFO(this->get_logger(),
                "[augment] new line landmark: rho=%.3f theta=%.3f (after %d hits) | "
                "n_lm=%d state_dim=%d",
                c.rho, c.theta, c.hits,
                ekf_->nLandmarks2(), ekf_->stateDim());
            line_candidates_.erase(line_candidates_.begin() + best);
        }
    }

    // -------------------- Corner landmark SLAM --------------------
    static constexpr size_t MAX_CORNER_LANDMARKS = 20;
    static constexpr double CONFIRM_XY_TOL_M     = 0.10;

    struct CornerCandidate {
        double xw;
        double yw;
        int    hits;
        int    last_seen_scan;
    };
    std::vector<CornerCandidate> corner_candidates_;

    void runCornerSlamUpdate(const std::vector<ekf_slam::CornerObs>& corners) {
        if (!scan_to_map_initialized_) return;

        bool diag_this_scan = false;
        double now_s = this->now().seconds();
        if (now_s - last_corner_diag_s_ >= 1.0) {
            last_corner_diag_s_ = now_s;
            diag_this_scan      = true;
        }

        for (size_t obs_i = 0; obs_i < corners.size(); ++obs_i) {
            const auto& obs = corners[obs_i];
            const double cx_obs = obs.x;
            const double cy_obs = obs.y;

            const int n_lm      = ekf_->nLandmarks2();
            const int n_corners = ekf_->nCorners();
            int   best_j   = -1;
            double best_d2 = std::numeric_limits<double>::infinity();
            Eigen::Vector2d best_z_pred(0, 0);
            Eigen::Vector2d best_nu(0, 0);

            for (int k = 0; k < n_lm; ++k) {
                if (ekf_->landmarkKind(k) != ekf_slam::LandmarkKind::Corner) continue;
                int idx = 4 + 2 * k;
                Eigen::Vector2d z_pred;
                Eigen::MatrixXd H;
                ekf_->predictCornerObservation(idx, z_pred, H);
                Eigen::Vector2d nu;
                nu(0) = cx_obs - z_pred(0);
                nu(1) = cy_obs - z_pred(1);
                Eigen::Matrix2d S = H * ekf_->getCovariance() * H.transpose() + Q_corner_;
                double d2 = nu.transpose() * S.inverse() * nu;
                if (d2 < best_d2) {
                    best_d2     = d2;
                    best_j      = idx;
                    best_z_pred = z_pred;
                    best_nu     = nu;
                }
            }

            if (diag_this_scan && obs_i == 0) {
                if (best_j >= 0) {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA-c] obs=(cx=%.3f cy=%.3f) | nearest_lm idx=%d "
                        "z_pred=(%.3f, %.3f) innov=(%.3f, %.3f) d2=%.2f "
                        "(gate=%.2f) | n_corners=%d candidates=%zu",
                        cx_obs, cy_obs, best_j,
                        best_z_pred(0), best_z_pred(1),
                        best_nu(0), best_nu(1), best_d2,
                        CHI2_GATE_2DOF, n_corners, corner_candidates_.size());
                } else {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA-c] obs=(cx=%.3f cy=%.3f) | no existing corners | "
                        "candidates=%zu",
                        cx_obs, cy_obs, corner_candidates_.size());
                }
            }

            if (best_j >= 0 && best_d2 < CHI2_GATE_2DOF) {
                bool applied = ekf_->updateCornerLandmark(best_j, cx_obs, cy_obs, Q_corner_);
                if (!applied) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                        1000,
                        "Corner update skipped: S near-singular for landmark idx=%d",
                        best_j);
                }
            } else if (n_corners < static_cast<int>(MAX_CORNER_LANDMARKS)) {
                tryConfirmCornerAndAugment(obs.xw, obs.yw);
            }
        }

        for (auto it = corner_candidates_.begin(); it != corner_candidates_.end();) {
            if (scans_seen_ - it->last_seen_scan > CANDIDATE_TTL) {
                it = corner_candidates_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void tryConfirmCornerAndAugment(double xw, double yw) {
        // Match candidates in WORLD frame (corner viz + extractor return both).
        // Augment uses the robot-frame value derived back from the world point
        // and current pose; here we just track world-frame for stability.
        int best = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < corner_candidates_.size(); ++i) {
            double dxw = std::abs(xw - corner_candidates_[i].xw);
            double dyw = std::abs(yw - corner_candidates_[i].yw);
            if (dxw < CONFIRM_XY_TOL_M && dyw < CONFIRM_XY_TOL_M) {
                double d = dxw + dyw;
                if (d < best_dist) {
                    best_dist = d;
                    best      = static_cast<int>(i);
                }
            }
        }
        if (best < 0) {
            corner_candidates_.push_back({xw, yw, 1, static_cast<int>(scans_seen_)});
            return;
        }
        auto& c = corner_candidates_[best];
        if (c.last_seen_scan == static_cast<int>(scans_seen_)) return;
        c.hits          += 1;
        c.last_seen_scan = static_cast<int>(scans_seen_);
        c.xw             = 0.5 * (c.xw + xw);
        c.yw             = 0.5 * (c.yw + yw);
        if (c.hits >= N_CONFIRM) {
            // Convert the confirmed world-frame candidate back to robot frame
            // for the augment, since augmentCornerFromObservation expects
            // robot-frame inputs.
            Eigen::Vector4d pose = ekf_->getPose();
            double psi = pose(3);
            double cs  = std::cos(-psi);
            double sn  = std::sin(-psi);
            double dx  = c.xw - pose(0);
            double dy  = c.yw - pose(1);
            double cx_r = cs * dx - sn * dy;
            double cy_r = sn * dx + cs * dy;
            ekf_->augmentCornerFromObservation(cx_r, cy_r, Q_corner_);
            RCLCPP_INFO(this->get_logger(),
                "[augment] new corner landmark: world=(%.3f, %.3f) (after %d hits) | "
                "n_corners=%d state_dim=%d",
                c.xw, c.yw, c.hits,
                ekf_->nCorners(), ekf_->stateDim());
            corner_candidates_.erase(corner_candidates_.begin() + best);
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
            if (ekf_->landmarkKind(k) != ekf_slam::LandmarkKind::Line) continue;
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

    /// Render every persisted corner landmark in the EKF state.
    void publishStateLandmarkCorners() {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = this->now();
        del.action          = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(del);

        const Eigen::VectorXd& mu = ekf_->getState();
        int n_lm = ekf_->nLandmarks2();
        for (int k = 0; k < n_lm; ++k) {
            if (ekf_->landmarkKind(k) != ekf_slam::LandmarkKind::Corner) continue;
            int idx = 4 + 2 * k;

            visualization_msgs::msg::Marker m;
            m.header.frame_id    = "map";
            m.header.stamp       = this->now();
            m.ns                 = "ekf_slam_landmark_corners";
            m.id                 = k;
            m.type               = visualization_msgs::msg::Marker::CUBE;
            m.action             = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x    = mu(idx);
            m.pose.position.y    = mu(idx + 1);
            m.pose.position.z    = 0.0;
            m.pose.orientation.w = 1.0;
            m.scale.x            = 0.12;
            m.scale.y            = 0.12;
            m.scale.z            = 0.12;
            m.color.r            = 0.95f;
            m.color.g            = 0.85f;
            m.color.b            = 0.1f;
            m.color.a            = 1.0f;
            arr.markers.push_back(m);
        }
        landmark_corners_pub_->publish(arr);
    }

    void logSlamState(const std::vector<ekf_slam::LineObs>& lines,
                      bool ran_scan_to_map) {
        int n_lines   = ekf_->nLines();
        int n_corners = ekf_->nCorners();
        int dim       = ekf_->stateDim();
        const char* sm = ran_scan_to_map ? "scan2map=on" : "scan2map=skipped";
        if (lines.empty()) {
            RCLCPP_INFO(this->get_logger(),
                "Line landmarks: %d | Corner landmarks: %d | State dim: %d | %s | "
                "(no line obs)",
                n_lines, n_corners, dim, sm);
        } else {
            const auto& last = lines.back();
            RCLCPP_INFO(this->get_logger(),
                "Line landmarks: %d | Corner landmarks: %d | State dim: %d | %s | "
                "Last line: rho=%.3f theta=%.3f",
                n_lines, n_corners, dim, sm, last.rho, last.theta);
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
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr   map_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr   pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_cov_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr lines_dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr corners_dbg_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_lines_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_corners_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    ekf_slam::LineExtractor line_extractor_;
    Eigen::Matrix2d         Q_line_;
    Eigen::Matrix2d         Q_corner_;

    double last_odom_time_;

    // scan-to-map state (primary correction)
    Eigen::Vector4d prev_scan_pose_     = Eigen::Vector4d::Zero();
    bool            prev_scan_pose_set_ = false;

    nav_msgs::msg::OccupancyGrid::SharedPtr current_map_;
    bool                                    map_received_ = false;
    std::mutex                              map_mutex_;

    // One-shot diagnostic for the K-dim verification log.
    bool logged_first_update_ = false;

    // Latch: line landmark augment/update is gated until the first time
    // scan-to-map successfully runs and corrects the pose.
    bool scan_to_map_initialized_ = false;

    // Scan counter (for confirmation-buffer TTL and start-of-flight diag).
    unsigned long scans_seen_ = 0;
    // Throttle for the per-observation Mahalanobis diag (~1 Hz).
    double last_diag_s_        = 0.0;
    double last_corner_diag_s_ = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
