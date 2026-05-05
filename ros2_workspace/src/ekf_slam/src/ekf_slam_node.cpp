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
#include "ekf_slam/landmark_filter.hpp"
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

        // Separate landmark filter — receives pose from EKFCore as a const
        // input, never modifies pose. See landmark_filter.hpp for the
        // architectural rationale.
        landmark_filter_ = std::make_unique<ekf_slam::LandmarkFilter>();

        // Q_line: measurement noise for (rho, theta) line observations in robot frame.
        // Inflated heavily from the Stage-2 spec (diag(0.04, 0.02)) so the EKF
        // trusts landmarks much less and the pose only gets nudged a little
        // per update. Scan-to-map remains the dominant pose anchor; lines
        // contribute correlation structure without dragging pose around.
        Q_line_ = Eigen::Matrix2d::Zero();
        Q_line_(0, 0) = 0.5;
        Q_line_(1, 1) = 0.2;

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
            "EKF-SLAM two-filter init: 4-state pose EKF (scan-to-map + odom yaw + "
            "down-range z) + separate LandmarkFilter for line landmarks "
            "(max %d). Corners derived geometrically from line pairs.",
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
        last_cmd_vz_ = msg->linear.z;
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
        // Threshold reverted to v1's value (50). Lowering it to 15 made
        // scan-to-map fire against a sparse map where nearest-occupied-cell
        // matching is unreliable — the resulting (dx, dy) observations were
        // noisy enough to drag pose around. Holding scan-to-map back until
        // the map is dense matches v1's known-good 3.2 cm landing behavior.
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

        // Innovation clipping: cap the per-step correction magnitude so a
        // single bad match (e.g. beam matched to a phantom cell from an
        // earlier drifted pose) cannot snap the EKF pose by more than
        // INNOVATION_CAP_M in one tick. With 2 beams frequently reading
        // inf this is the standard robustness trick — keep scan-to-map
        // firing every scan, but bound how much damage any one match can
        // do. The EKF will need a few ticks to fully accept a real, large
        // correction, but the cumulative effect is fine since scans are
        // 10 Hz.
        constexpr double INNOVATION_CAP_M = 0.08;
        double corr_x = sum_dx / valid_beams;
        double corr_y = sum_dy / valid_beams;
        double corr_mag = std::hypot(corr_x, corr_y);
        if (corr_mag > INNOVATION_CAP_M) {
            double scale = INNOVATION_CAP_M / corr_mag;
            corr_x *= scale;
            corr_y *= scale;
        }

        result.dx            = robot_x + corr_x;
        result.dy            = robot_y + corr_y;
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

        // Descent gate: when nav has commanded descent (vz < -0.05) AND we're
        // below 0.4 m, freeze all corrections — scan-to-map AND landmark
        // updates. The EKF still runs the prediction step from /odom so z and
        // yaw track, but xy stays locked at whatever it was when nav decided
        // "goal reached" and entered LANDING. This eliminates the 5–10 cm
        // sideways jitter during straight-down descent that was causing nav
        // to overshoot the pad.
        bool in_descent = (last_cmd_vz_ < -0.05) && (pre_pose(2) < 0.4);
        if (in_descent) {
            scans_seen_++;
            return;
        }

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
                // Guard A: scan-to-map is anchoring the pose again. Reset the
                // dropout counter and unpause landmarks if they were paused.
                sm_failed_streak_ = 0;
                if (landmarks_paused_) {
                    landmarks_paused_ = false;
                    RCLCPP_INFO(this->get_logger(),
                        "[guard] scan-to-map recovered — landmark augment/update resumed.");
                }
                RCLCPP_DEBUG(this->get_logger(),
                    "scan-to-map abs=(%.3f, %.3f, %.3f) q=%.3f",
                    sm.dx, sm.dy, sm.dtheta, sm.match_quality);
            } else if (scan_to_map_initialized_) {
                // Guard A: scan-to-map was working, now it's not. Count
                // consecutive failures; once we cross the threshold pause
                // landmark augment/update so the cascade described in the
                // last flight log can't happen (drift -> wrong-anchored
                // augments -> failed DA -> spawn more wrong landmarks).
                sm_failed_streak_++;
                if (sm_failed_streak_ >= SM_DROPOUT_THRESHOLD && !landmarks_paused_) {
                    landmarks_paused_ = true;
                    RCLCPP_WARN(this->get_logger(),
                        "[guard] scan-to-map dropped out for %d consecutive valid-gate "
                        "scans — pausing landmark augment/update until it recovers. "
                        "Predict-only EKF stays running.",
                        sm_failed_streak_);
                }
            }
            prev_scan_pose_     = ekf_->getPose();
            prev_scan_pose_set_ = true;
        }

        // Periodic diagnostic so we can tell whether scan2map=skipped means
        // (a) translation gate held, (b) gate passed but no map / not enough
        // occupied cells / not enough valid beams, or (c) map sub never fired.
        // Fires for the first 5 scans, then once per second for the rest of
        // the flight so we can watch the latch progression live.
        double now_s = this->now().seconds();
        bool fire_diag = (scans_seen_ < 5) ||
                         (now_s - last_scan_diag_s_ >= 1.0);
        if (fire_diag) {
            last_scan_diag_s_ = now_s;
            int occupied = 0;
            {
                std::lock_guard<std::mutex> lock(map_mutex_);
                if (current_map_) {
                    for (auto& cell : current_map_->data) if (cell == 100) occupied++;
                }
            }
            RCLCPP_INFO(this->get_logger(),
                "[scanCB diag #%lu] map_received=%d occupied_cells=%d trans_dist=%.4fm "
                "gate_passed=%d sm_valid=%d sm_q=%.2f ran_sm=%d sm_init=%d",
                scans_seen_,
                static_cast<int>(map_received_),
                occupied,
                trans_dist,
                static_cast<int>(gate_passed),
                static_cast<int>(sm_valid),
                sm_quality,
                static_cast<int>(ran_scan_to_map),
                static_cast<int>(scan_to_map_initialized_));
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
        // Corner observations are no longer added to any filter state;
        // corners are derived geometrically from the line landmarks at
        // visualization time (see publishStateLandmarkCorners).
        (void)corners;

        publishStateLandmarkLines();
        publishStateLandmarkCorners();
        logSlamState(lines, ran_scan_to_map);
    }

    // Confirmation buffer for new-landmark spawning. An observation is only
    // augmented to the EKF state once we have seen N_CONFIRM matches within
    // the confirm-gate of each other across distinct scans. Defends against
    // transient front-end fits running away with the state.
    static constexpr int    N_CONFIRM        = 8;
    static constexpr double CONFIRM_RHO_TOL  = 0.10;  // m
    static constexpr double CONFIRM_THETA_TOL = 0.10;  // rad
    static constexpr int    CANDIDATE_TTL    = 30;    // scans without re-obs before forget

    // Robot-frame rho bounds. Below RHO_MIN are near-origin garbage fits;
    // above RHO_MAX are outside the room and almost certainly extractor
    // noise (e.g. from a single bucket with two beam endpoints far apart).
    static constexpr double RHO_MIN_M = 0.15;
    static constexpr double RHO_MAX_M = 2.5;

    // Guard B: room half-extent (the simulated room is roughly 4x4m centred
    // on origin). Augment is rejected if the would-be world landmark is
    // outside [-ROOM_HALF_M-ROOM_MARGIN_M, +ROOM_HALF_M+ROOM_MARGIN_M].
    static constexpr double ROOM_HALF_M   = 2.0;
    static constexpr double ROOM_MARGIN_M = 0.5;

    struct LineCandidate {
        // Stored in WORLD frame. Robot-frame (rho_r, theta_r) shifts as the
        // drone moves/yaws even when observing the same wall, so confirming
        // in robot frame would never accumulate enough hits. Converting to
        // world frame at observation time matches the corner candidate
        // approach (which has always been world-frame).
        double rho_w;
        double theta_w;
        int    hits;
        int    last_seen_scan;
    };
    std::vector<LineCandidate> line_candidates_;

    /// For each observed line: Mahalanobis-gate against existing landmarks;
    /// update if matched. If unmatched, route through the confirmation buffer
    /// and only augment once we have N_CONFIRM consistent observations.
    void runLineSlamUpdate(const std::vector<ekf_slam::LineObs>& lines) {
        // Two-filter architecture: landmark updates run on the separate
        // LandmarkFilter and never touch the pose EKF. Pose comes in as a
        // const input every scan.

        // Per-scan diagnostic throttle (~1 Hz).
        bool diag_this_scan = false;
        double now_s = this->now().seconds();
        if (now_s - last_diag_s_ >= 1.0) {
            last_diag_s_   = now_s;
            diag_this_scan = true;
        }

        const Eigen::Vector4d pose = ekf_->getPose();

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

            const int n_lm = landmark_filter_->nLandmarks();
            int   best_k   = -1;
            double best_d2 = std::numeric_limits<double>::infinity();
            Eigen::Vector2d best_z_pred(0, 0);
            Eigen::Vector2d best_nu(0, 0);

            // DA over the landmark filter's covariance only — no coupling
            // with pose covariance. H_lm is 2x2; S = H_lm * Sigma_block_kk
            // * H_lm^T + Q_line.
            const Eigen::MatrixXd& Sigma_lm = landmark_filter_->getCovariance();
            for (int k = 0; k < n_lm; ++k) {
                Eigen::Vector2d z_pred;
                Eigen::Matrix2d H_lm;
                landmark_filter_->predictLineObservation(pose, k, z_pred, H_lm);
                Eigen::Vector2d nu;
                nu(0) = rho_obs - z_pred(0);
                nu(1) = ekf_slam::EKFCore::normalizeAngle(theta_obs - z_pred(1));
                const int row = 2 * k;
                Eigen::Matrix2d Sigma_kk = Sigma_lm.block<2, 2>(row, row);
                Eigen::Matrix2d S = H_lm * Sigma_kk * H_lm.transpose() + Q_line_;
                double d2 = nu.transpose() * S.inverse() * nu;
                if (d2 < best_d2) {
                    best_d2     = d2;
                    best_k      = k;
                    best_z_pred = z_pred;
                    best_nu     = nu;
                }
            }

            if (diag_this_scan && obs_i == 0) {
                if (best_k >= 0) {
                    RCLCPP_INFO(this->get_logger(),
                        "[DA] obs=(rho=%.3f th=%.3f) | nearest_lm k=%d "
                        "z_pred=(%.3f, %.3f) innov=(%.3f, %.3f) d2=%.2f "
                        "(gate=%.2f) | n_lm=%d candidates=%zu",
                        rho_obs, theta_obs, best_k,
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

            if (best_k >= 0 && best_d2 < CHI2_GATE_2DOF) {
                bool applied = landmark_filter_->updateLine(
                    pose, best_k, rho_obs, theta_obs, Q_line_);
                if (!applied) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(),
                        1000,
                        "Line update skipped: S near-singular for landmark k=%d "
                        "(landmark covariance has shrunk too far).", best_k);
                } else if (!logged_first_update_) {
                    RCLCPP_INFO(this->get_logger(),
                        "[verify] first line landmark update applied. "
                        "landmark state dim = %d. Pose EKF and landmark filter "
                        "are independent — pose untouched by this update.",
                        landmark_filter_->stateDim());
                    logged_first_update_ = true;
                }
            } else if (n_lm < static_cast<int>(MAX_LINE_LANDMARKS)) {
                // Route through the confirmation buffer.
                tryConfirmLineAndAugment(rho_obs, theta_obs);
            }
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
        // Convert this scan's robot-frame observation into world frame so the
        // candidate buffer can match across drone motion.
        Eigen::Vector4d pose = ekf_->getPose();
        double psi      = pose(3);
        double theta_w  = ekf_slam::EKFCore::normalizeAngle(theta_obs + psi);
        double rho_w    = rho_obs +
                          pose(0) * std::cos(theta_w) +
                          pose(1) * std::sin(theta_w);

        // Find the nearest candidate within the confirm tolerance.
        int best = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < line_candidates_.size(); ++i) {
            double drho = std::abs(rho_w - line_candidates_[i].rho_w);
            double dth  = std::abs(ekf_slam::EKFCore::normalizeAngle(
                              theta_w - line_candidates_[i].theta_w));
            if (drho < CONFIRM_RHO_TOL && dth < CONFIRM_THETA_TOL) {
                double d = drho + dth;
                if (d < best_dist) {
                    best_dist = d;
                    best      = static_cast<int>(i);
                }
            }
        }
        if (best < 0) {
            line_candidates_.push_back({rho_w, theta_w, 1,
                                        static_cast<int>(scans_seen_)});
            return;
        }
        auto& c = line_candidates_[best];
        if (c.last_seen_scan == static_cast<int>(scans_seen_)) return;
        c.hits          += 1;
        c.last_seen_scan = static_cast<int>(scans_seen_);
        c.rho_w          = 0.5 * (c.rho_w + rho_w);
        c.theta_w        = 0.5 * (c.theta_w + theta_w);
        if (c.hits >= N_CONFIRM) {
            // Guard B: reject candidates whose world line is outside the
            // simulated room.
            if (std::abs(c.rho_w) > ROOM_HALF_M + ROOM_MARGIN_M) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "[guard] rejecting line candidate: |rho_w|=%.2f > %.2f "
                    "(theta_w=%.2f)", std::abs(c.rho_w),
                    ROOM_HALF_M + ROOM_MARGIN_M, c.theta_w);
                line_candidates_.erase(line_candidates_.begin() + best);
                return;
            }
            // Convert confirmed world line back to robot frame for the augment
            // (augmentLine expects robot-frame inputs and the current pose).
            Eigen::Vector4d aug_pose = ekf_->getPose();
            double aug_psi    = aug_pose(3);
            double rho_r_aug  = c.rho_w
                                - aug_pose(0) * std::cos(c.theta_w)
                                - aug_pose(1) * std::sin(c.theta_w);
            double theta_r_aug = ekf_slam::EKFCore::normalizeAngle(c.theta_w - aug_psi);
            landmark_filter_->augmentLine(aug_pose, rho_r_aug, theta_r_aug, Q_line_);
            RCLCPP_INFO(this->get_logger(),
                "[augment] new line landmark: world (rho=%.3f theta=%.3f) "
                "(after %d hits) | n_lm=%d state_dim=%d",
                c.rho_w, c.theta_w, c.hits,
                landmark_filter_->nLandmarks(), landmark_filter_->stateDim());
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

    // Corner SLAM removed: corners are derived geometrically from the line
    // landmarks at visualization time (publishStateLandmarkCorners). No
    // corner state, no corner Kalman update, no corner candidate buffer.


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

        const Eigen::VectorXd& mu = landmark_filter_->getState();
        int n_lm = landmark_filter_->nLandmarks();
        for (int k = 0; k < n_lm; ++k) {
            int idx = 2 * k;
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

    /// Render corners DERIVED geometrically from the line landmarks already
    /// in the EKF state. Corners are NOT part of the state — they're a pure
    /// post-processing visualization. For each pair of stored lines whose
    /// normals are near-perpendicular (|cos(θ1-θ2)| < 0.3), solve the 2x2
    /// system [cosθ1 sinθ1; cosθ2 sinθ2] [cx; cy] = [ρ1; ρ2] for the
    /// intersection point.
    void publishStateLandmarkCorners() {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker del;
        del.header.frame_id = "map";
        del.header.stamp    = this->now();
        del.action          = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(del);

        const Eigen::VectorXd& mu = landmark_filter_->getState();
        int n_lm = landmark_filter_->nLandmarks();

        struct LineLm { double rho; double theta; };
        std::vector<LineLm> lines;
        lines.reserve(n_lm);
        for (int k = 0; k < n_lm; ++k) {
            int idx = 2 * k;
            lines.push_back({mu(idx), mu(idx + 1)});
        }

        // Bound: anything outside this is past the room walls — skip it.
        constexpr double ROOM_HALF_M   = 2.0;
        constexpr double ROOM_MARGIN_M = 0.5;
        constexpr double LIMIT = ROOM_HALF_M + ROOM_MARGIN_M;

        int marker_id = 0;
        for (size_t i = 0; i < lines.size(); ++i) {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                double t1 = lines[i].theta;
                double t2 = lines[j].theta;
                // Perpendicularity gate: |cos(t1 - t2)| < 0.3 → angle within
                // ~17° of 90°.
                if (std::abs(std::cos(t1 - t2)) >= 0.3) continue;

                // Solve the 2x2 system.
                Eigen::Matrix2d A;
                A << std::cos(t1), std::sin(t1),
                     std::cos(t2), std::sin(t2);
                Eigen::Vector2d b(lines[i].rho, lines[j].rho);
                double det = A.determinant();
                if (std::abs(det) < 1e-6) continue;
                Eigen::Vector2d c = A.inverse() * b;
                double cx = c(0);
                double cy = c(1);
                if (std::abs(cx) > LIMIT || std::abs(cy) > LIMIT) continue;

                visualization_msgs::msg::Marker m;
                m.header.frame_id    = "map";
                m.header.stamp       = this->now();
                m.ns                 = "ekf_slam_derived_corners";
                m.id                 = marker_id++;
                m.type               = visualization_msgs::msg::Marker::CUBE;
                m.action             = visualization_msgs::msg::Marker::ADD;
                m.pose.position.x    = cx;
                m.pose.position.y    = cy;
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
        }
        landmark_corners_pub_->publish(arr);
    }

    void logSlamState(const std::vector<ekf_slam::LineObs>& lines,
                      bool ran_scan_to_map) {
        // Two-filter architecture: pose EKF is fixed 4-state; landmark filter
        // owns the (rho, theta) line landmarks. Corners are derived
        // geometrically and have no state.
        int n_lines   = landmark_filter_->nLandmarks();
        int n_corners = 0;
        int dim       = 4 + landmark_filter_->stateDim();
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

    std::unique_ptr<ekf_slam::EKFCore>        ekf_;
    std::unique_ptr<ekf_slam::LandmarkFilter> landmark_filter_;

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

    // Guard A: dropout-watchdog state. Once scan-to-map has been working,
    // count consecutive valid-gate scans where it returned invalid. After
    // SM_DROPOUT_THRESHOLD failures pause landmark augment/update; resume
    // when scan-to-map produces a valid match again.
    static constexpr int SM_DROPOUT_THRESHOLD = 30;  // ~3 s at 10 Hz
    int  sm_failed_streak_ = 0;
    bool landmarks_paused_ = false;

    // Cached commanded vz from /cmd_vel — used by the descent gate in
    // scanCallback to freeze xy corrections during straight-down landings.
    double last_cmd_vz_ = 0.0;

    // Scan counter (for confirmation-buffer TTL and start-of-flight diag).
    unsigned long scans_seen_ = 0;
    // Throttle for the per-observation Mahalanobis diag (~1 Hz).
    double last_diag_s_        = 0.0;
    double last_corner_diag_s_ = 0.0;
    // Throttle for the scan-callback / scan-to-map state diag (~1 Hz).
    double last_scan_diag_s_   = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFSlamNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
