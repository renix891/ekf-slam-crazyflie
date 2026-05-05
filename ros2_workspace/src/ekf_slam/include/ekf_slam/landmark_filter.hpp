#ifndef EKF_SLAM_LANDMARK_FILTER_HPP
#define EKF_SLAM_LANDMARK_FILTER_HPP

#include <Eigen/Dense>

namespace ekf_slam {

/**
 * @brief Bayesian filter over a (rho, theta) line-landmark map.
 *
 * State vector: mu_lm = [rho_1, theta_1, rho_2, theta_2, ...]   (2N x 1)
 * Covariance:   Sigma_lm                                        (2N x 2N)
 *
 * Pose is supplied as a CONST INPUT to every method (Vector4d
 * [x, y, z, theta]). The filter never modifies the pose and never owns
 * pose covariance — those live in the separate EKFCore. Every observation
 * (rho_obs, theta_obs) is in the robot frame; the filter converts to world
 * frame for storage using the supplied pose, treating pose as known.
 *
 * This decouples landmark mapping from pose estimation: any numerical
 * trouble in landmark Jacobians, augmentation, or update math is bounded
 * to this filter's state and cannot leak back into pose.
 */
class LandmarkFilter {
public:
    LandmarkFilter() = default;

    /**
     * @brief Initialize a new line landmark in world frame from a robot-frame
     *        observation. Returns the landmark index (0-based) of the new
     *        block in the state vector.
     *
     * @param pose         Current robot pose (read-only).
     * @param rho_obs_r    Observed perpendicular distance in robot frame [m].
     * @param theta_obs_r  Observed line-normal bearing in robot frame [rad].
     * @param R_line       2x2 measurement noise on (rho_obs, theta_obs).
     */
    int augmentLine(const Eigen::Vector4d& pose,
                    double rho_obs_r, double theta_obs_r,
                    const Eigen::Matrix2d& R_line);

    /**
     * @brief Apply a Kalman correction to landmark `landmark_idx` from a
     *        robot-frame observation. Pose is treated as a constant. Only the
     *        landmark block (rows/cols 2*idx..2*idx+1) and its cross-
     *        correlations with other landmarks are updated.
     *
     * @return true if the update was applied; false if the innovation
     *         covariance was numerically ill-conditioned and the update was
     *         skipped (filter stays bounded).
     */
    bool updateLine(const Eigen::Vector4d& pose,
                    int landmark_idx,
                    double rho_obs_r, double theta_obs_r,
                    const Eigen::Matrix2d& R_line);

    /**
     * @brief Predict the (rho_obs_r, theta_obs_r) measurement that would be
     *        seen for landmark `landmark_idx` given pose. H_lm is the 2x2
     *        Jacobian over the landmark block only (pose Jacobian is NOT
     *        returned — pose is constant from this filter's perspective).
     */
    void predictLineObservation(const Eigen::Vector4d& pose,
                                int landmark_idx,
                                Eigen::Vector2d& z_pred,
                                Eigen::Matrix2d& H_lm) const;

    int  nLandmarks() const { return static_cast<int>(mu_lm_.size() / 2); }
    int  stateDim()   const { return static_cast<int>(mu_lm_.size()); }

    const Eigen::VectorXd& getState()      const { return mu_lm_; }
    const Eigen::MatrixXd& getCovariance() const { return Sigma_lm_; }

private:
    Eigen::VectorXd mu_lm_;       // [rho_1, theta_1, rho_2, theta_2, ...]
    Eigen::MatrixXd Sigma_lm_;    // 2N x 2N

    // Wrap angles after a state update (column index 1, 3, 5, ... in mu_lm_).
    void normalizeStateAngles();
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_LANDMARK_FILTER_HPP
