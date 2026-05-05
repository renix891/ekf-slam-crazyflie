#ifndef EKF_SLAM_LINE_EXTRACTOR_HPP
#define EKF_SLAM_LINE_EXTRACTOR_HPP

#include <Eigen/Dense>
#include <array>
#include <deque>
#include <map>
#include <vector>

namespace ekf_slam {

/// Single line observation in the robot frame.
/// rho = signed perpendicular distance from robot to the line.
/// theta = bearing of the line's outward normal, measured in the robot frame.
struct LineObs {
    double          rho;
    double          theta;
    Eigen::Matrix2d cov;        ///< 2x2 covariance over (rho, theta)
    int             n_points;   ///< support count (informational)
    int             bucket_deg; ///< world-bearing bucket of this fit (informational, debug)
    // World-frame fit (debug / visualization only).
    double          rho_w;
    double          theta_w;
};

/// Corner observation in the robot frame.
struct CornerObs {
    double          x;          ///< robot-frame corner x
    double          y;          ///< robot-frame corner y
    Eigen::Matrix2d cov;        ///< 2x2 covariance over (x, y)
    int             bucket1_deg;
    int             bucket2_deg;
    // World-frame coordinates (debug / visualization only).
    double          xw;
    double          yw;
};

/**
 * @brief Buffer-driven line / corner extractor for the 4-beam multiranger.
 *
 * Stores recent beam endpoints in WORLD coordinates, bucketed by world bearing
 * (rounded to BUCKET_DEG steps) so each bucket accumulates samples from
 * whichever beam was pointing in that direction. This survives drone yaw,
 * unlike bucketing by beam index.
 *
 * extract() runs gap-split + OLS line fit per bucket, then transforms the
 * resulting (rho_w, theta_w) lines into the robot frame using the supplied
 * current pose. Corners are intersections of adjacent buckets' lines, also
 * returned in the robot frame.
 */
class LineExtractor {
public:
    struct Params {
        size_t buffer_scans  = 30;      ///< rolling buffer length
        int    bucket_deg    = 15;      ///< world-bearing bucket size
        double gap_thresh    = 0.15;    ///< split if consecutive endpoints exceed this (m)
        size_t min_points    = 5;       ///< minimum points to fit a segment
        double max_range     = 3.49;    ///< treat ranges >= this as "no return"
        double min_range     = 0.05;    ///< ranges below this are sensor floor
        // Diagonal-constant landmark covariances (per spec).
        double line_var_rho   = 0.02;
        double line_var_theta = 0.01;
        double corner_var_xy  = 0.04;   ///< per-axis variance for corner xy
        // Corner gating
        double corner_max_dist = 3.0;   ///< reject corners further than this from robot (m)
        double corner_perp_cos = 0.3;   ///< |cos(theta1-theta2)| must be < this
    };

    LineExtractor();
    explicit LineExtractor(const Params& p);

    /// Push a fresh scan. Endpoints are stored in world coordinates.
    /// @param ranges_b   per-beam ranges (m), order must match bearings_b
    /// @param bearings_b per-beam *body* bearing (rad), e.g. {pi, -pi/2, 0, pi/2}
    /// @param x, y, yaw  robot pose at time of scan (world frame)
    /// @param valid_mask if any element is false, that beam is skipped
    void addScan(double x, double y, double yaw,
                 const std::array<double, 4>& ranges_b,
                 const std::array<double, 4>& bearings_b);

    /// Extract line + corner observations expressed in the robot frame at the
    /// supplied current pose. Empties intermediate state but keeps the buffer.
    void extract(double cur_x, double cur_y, double cur_yaw,
                 std::vector<LineObs>& out_lines,
                 std::vector<CornerObs>& out_corners) const;

    const Params& params() const { return params_; }

    /// Read-only access for debug printing: returns {bucket_deg -> point_count}.
    std::map<int, size_t> bucketSizes() const {
        std::map<int, size_t> out;
        for (const auto& kv : buckets_) out[kv.first] = kv.second.size();
        return out;
    }

    /// Total number of endpoints currently buffered across all buckets.
    size_t totalEndpoints() const {
        size_t n = 0;
        for (const auto& kv : buckets_) n += kv.second.size();
        return n;
    }

    /// Latest scan sequence number (monotonically increases by 1 per addScan).
    double seq() const { return seq_; }

private:
    struct Endpoint {
        double wx;     ///< world x
        double wy;     ///< world y
        double t_seq;  ///< monotonically increasing sequence id (for ordering within a bucket)
    };

    Params params_;
    // bucket_deg -> rolling list of recent endpoints in that bearing.
    std::map<int, std::deque<Endpoint>> buckets_;
    // Sequence counter so we can keep the buckets time-ordered without a clock.
    double seq_ = 0.0;
    // Per-bucket cap so a long-lived bucket doesn't grow unbounded.
    size_t bucket_cap_ = 120;  // ~30 scans * 4 beams of headroom

    // Helpers
    static int  bucketOfWorldBearing(double bearing_rad, int bucket_deg);
    static bool fitLineWorld(const std::vector<Eigen::Vector2d>& pts,
                             double& rho_w, double& theta_w);
    static void worldLineToRobot(double rho_w, double theta_w,
                                 double rx, double ry, double ryaw,
                                 double& rho_r, double& theta_r);
    static double normalizeAngle(double a);
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_LINE_EXTRACTOR_HPP
