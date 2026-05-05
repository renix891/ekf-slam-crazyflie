#ifndef EKF_SLAM_EKF_CORE_HPP
#define EKF_SLAM_EKF_CORE_HPP

#include <Eigen/Dense>

namespace ekf_slam {

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
     * @param landmark_idx  Index returned by augmentLineFromObservation (or
     *                      derived from data association).
     */
    void updateLineLandmark(int landmark_idx,
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

    Eigen::Vector4d  getPose() const;            // pose block (x, y, z, theta)
    Eigen::Matrix4d  getPoseCovariance() const;  // top-left 4x4 of Sigma_

    const Eigen::VectorXd& getState() const      { return mu_; }
    const Eigen::MatrixXd& getCovariance() const { return Sigma_; }
    Eigen::VectorXd&       mutableState()        { return mu_; }
    Eigen::MatrixXd&       mutableCovariance()   { return Sigma_; }

    int stateDim()    const { return static_cast<int>(mu_.size()); }
    int nLandmarks2() const { return (stateDim() - 4) / 2; }  // # of 2-D landmark blocks

    static double normalizeAngle(double angle);

private:
    Eigen::VectorXd mu_;     // dim = 4 + sum(landmark_block_sizes)
    Eigen::MatrixXd Sigma_;  // square, same dim as mu_

    Eigen::Matrix4d R_;            // pose-block process noise (4x4)
    Eigen::Matrix3d Q_scanmatch_;  // (x, y, theta) scan-match noise

    double commanded_vz_ = 0.0;

    Eigen::Matrix4d computePoseJacobian(double vx, double vy, double dt) const;
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_EKF_CORE_HPP
