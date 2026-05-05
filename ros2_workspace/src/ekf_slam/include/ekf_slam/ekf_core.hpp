#ifndef EKF_SLAM_EKF_CORE_HPP
#define EKF_SLAM_EKF_CORE_HPP

#include <Eigen/Dense>
#include <vector>

namespace ekf_slam {

/// Tag for each 2-D landmark block stored in mu_/Sigma_ so the post-update
/// normalization (which only wraps theta on line landmarks) can act on the
/// right blocks.
enum class LandmarkKind { Line, Corner };

/**
 * @brief EKF localization core with a growable state vector.
 *
 * Pose layout (always the first 4 entries of mu_):
 *   index 0 = x      (m, world)
 *   index 1 = y      (m, world)
 *   index 2 = z      (m, world; AGL on real flow-deck flights)
 *   index 3 = theta  (rad, yaw)
 *
 * Landmarks are appended after the pose by augment().  The pose block math is
 * unchanged from the fixed-size predecessor; updates expand their gain to
 * full-state dimension so the cross-covariance with landmarks gets updated
 * correctly once landmarks exist.
 */
class EKFCore {
public:
    EKFCore(const Eigen::Matrix4d& process_noise   = Eigen::Matrix4d::Identity() * 0.01,
            const Eigen::Matrix3d& scanmatch_noise = Eigen::Matrix3d::Identity() * 0.5);

    /// Predict step. Pose-only motion model; landmarks are static.
    void predict(double vx, double vy, double vz, double omega, double dt);

    /// Scan-match correction (absolute world-frame x, y, theta observation).
    void updateScanMatch(double dx, double dy, double dtheta, double match_quality);

    /// 1-D Kalman correction on theta.
    void updateYaw(double yaw_meas, double yaw_noise = 0.01);

    /// 1-D Kalman correction on z from a downward range source.
    void updateZ(double z_meas, double z_noise = 0.01);

    /// Latest commanded vz; drives the updateZ outlier gate.
    void setCommandedVz(double vz);

    /**
     * @brief Append a new landmark block to the state.
     *
     * @param new_state_block  k-vector of new state entries (e.g. [rho, theta] or [cx, cy]).
     * @param new_block_cov    k x k covariance for the new block.
     * @param cross_cov        k x stateDim() correlation between the new block and the
     *                         existing state. Pass a zero matrix if uncorrelated.
     * @return Index in mu_ at which the new block begins.
     */
    int augment(const Eigen::VectorXd& new_state_block,
                const Eigen::MatrixXd& new_block_cov,
                const Eigen::MatrixXd& cross_cov);

    /**
     * @brief Initialize a line landmark from a robot-frame observation.
     *
     * Inverts the (rho, theta) measurement model and appends the resulting
     * (rho_w, theta_w) world-frame line to the state, computing the proper
     * Jacobian-propagated cross-covariance with the existing pose so future
     * observations can correct both pose and landmark consistently.
     *
     * @return Index in mu_ at which (rho_w, theta_w) begins (always even, >=4).
     */
    int augmentLineFromObservation(double rho_obs_r, double theta_obs_r,
                                   const Eigen::Matrix2d& R_line);

    /**
     * @brief EKF correction from a robot-frame line observation against an
     *        existing line landmark.
     *
     * Skips the update if the innovation covariance S is near-singular
     * (det(S) < 1e-9 or ||S|| < 1e-6). Returns true on a real update,
     * false if the conditioning gate skipped it.
     *
     * @param landmark_idx  Index returned by augmentLineFromObservation (or
     *                      derived from data association).
     */
    bool updateLineLandmark(int landmark_idx,
                            double rho_obs_r, double theta_obs_r,
                            const Eigen::Matrix2d& R_line);

    /**
     * @brief Predicted robot-frame observation of an existing line landmark
     *        and the (m x stateDim) measurement Jacobian. Used by data
     *        association in the node.
     */
    void predictLineObservation(int landmark_idx,
                                Eigen::Vector2d& z_pred,
                                Eigen::MatrixXd& H) const;

    /**
     * @brief Initialize a corner landmark from a robot-frame (cx_r, cy_r)
     *        observation. Inverse model places the corner in world frame
     *        using the current pose; J_pose / J_obs propagate covariance.
     *        No canonicalization needed (corners are smooth Cartesian).
     *
     * @return Index in mu_ at which (cx_w, cy_w) begins.
     */
    int augmentCornerFromObservation(double cx_r, double cy_r,
                                     const Eigen::Matrix2d& R_corner);

    /**
     * @brief Predicted robot-frame observation of an existing corner landmark
     *        and the (m x stateDim) measurement Jacobian.
     */
    void predictCornerObservation(int landmark_idx,
                                  Eigen::Vector2d& z_pred,
                                  Eigen::MatrixXd& H) const;

    /**
     * @brief EKF correction from a robot-frame corner observation. Same
     *        S-conditioning gate as updateLineLandmark.
     */
    bool updateCornerLandmark(int landmark_idx,
                              double cx_r, double cy_r,
                              const Eigen::Matrix2d& R_corner);

    /// Kind of the k-th landmark block (0-indexed). Only valid for k in
    /// [0, nLandmarks2()). Used by updates to decide whether the second slot
    /// is an angle (line theta) or a Cartesian (corner cy).
    LandmarkKind landmarkKind(int k) const {
        return kinds_[static_cast<size_t>(k)];
    }

    Eigen::Vector4d  getPose() const;            // pose block (x, y, z, theta)
    Eigen::Matrix4d  getPoseCovariance() const;  // top-left 4x4 of Sigma_

    const Eigen::VectorXd& getState() const      { return mu_; }
    const Eigen::MatrixXd& getCovariance() const { return Sigma_; }
    Eigen::VectorXd&       mutableState()        { return mu_; }
    Eigen::MatrixXd&       mutableCovariance()   { return Sigma_; }

    int stateDim()    const { return static_cast<int>(mu_.size()); }
    int nLandmarks2() const { return (stateDim() - 4) / 2; }  // # of 2-D landmark blocks
    int nLines() const {
        int n = 0; for (auto k : kinds_) if (k == LandmarkKind::Line) n++; return n;
    }
    int nCorners() const {
        int n = 0; for (auto k : kinds_) if (k == LandmarkKind::Corner) n++; return n;
    }

    static double normalizeAngle(double angle);

private:
    Eigen::VectorXd mu_;     // dim = 4 + sum(landmark_block_sizes)
    Eigen::MatrixXd Sigma_;  // square, same dim as mu_
    std::vector<LandmarkKind> kinds_;  // tag for each 2-D landmark block

    Eigen::Matrix4d R_;            // pose-block process noise (4x4)
    Eigen::Matrix3d Q_scanmatch_;  // (x, y, theta) scan-match noise

    double commanded_vz_ = 0.0;

    Eigen::Matrix4d computePoseJacobian(double vx, double vy, double dt) const;

    // Re-canonicalize angle entries after a state update. Wraps theta on
    // line landmarks but leaves corner cy alone.
    void normalizeStateAngles();
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_EKF_CORE_HPP
