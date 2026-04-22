#ifndef EKF_SLAM_EKF_CORE_HPP
#define EKF_SLAM_EKF_CORE_HPP

#include <Eigen/Dense>
#include <vector>

namespace ekf_slam {

/**
 * @brief EKF-SLAM implementation for line-based features
 *
 * State vector: [x, y, theta, mx1, my1, mx2, my2, ...]
 *   - (x, y, theta): robot pose in world frame
 *   - (mxi, myi): landmark i position in world frame
 *
 * Motion model: constant velocity with noise
 *   x_k+1 = x_k + vx*cos(theta)*dt - vy*sin(theta)*dt
 *   y_k+1 = y_k + vx*sin(theta)*dt + vy*cos(theta)*dt
 *   theta_k+1 = theta_k + omega*dt
 *
 * Measurement model: range-bearing to landmarks
 *   rho = sqrt((mx - x)^2 + (my - y)^2)
 *   phi = atan2(my - y, mx - x) - theta
 */
class EKFCore {
public:
    /**
     * @brief Constructor
     * @param process_noise Process noise covariance (3x3, for x, y, theta)
     * @param measurement_noise Measurement noise covariance (2x2, for rho, theta)
     * @param mahalanobis_threshold Chi-squared threshold for data association (default 9.21 for 95% confidence)
     */
    EKFCore(const Eigen::Matrix3d& process_noise = Eigen::Matrix3d::Identity() * 0.01,
            const Eigen::Matrix2d& measurement_noise = Eigen::Matrix2d::Identity() * 0.05,
            double mahalanobis_threshold = 9.21);

    /**
     * @brief Prediction step using odometry
     * @param vx Linear velocity in body frame x-direction (m/s)
     * @param vy Linear velocity in body frame y-direction (m/s)
     * @param omega Angular velocity (rad/s)
     * @param dt Time step (s)
     */
    void predict(double vx, double vy, double omega, double dt);

    /**
     * @brief Correction step for a known landmark
     * @param rho Measured range to landmark (m)
     * @param theta_obs Measured bearing to landmark (rad, in body frame)
     * @param landmark_id Index of the landmark in state vector
     */
    void update(double rho, double theta_obs, int landmark_id);

    /**
     * @brief Add new landmark to state vector
     * @param rho Measured range to landmark (m)
     * @param theta_obs Measured bearing to landmark (rad, in body frame)
     * @return Index of newly added landmark
     */
    int addLandmark(double rho, double theta_obs);

    /**
     * @brief Associate measurement with existing landmarks using Mahalanobis distance gating
     * @param rho Measured range (m)
     * @param theta_obs Measured bearing (rad)
     * @return Landmark ID if match found, -1 if new landmark
     */
    int associateLandmark(double rho, double theta_obs);

    /**
     * @brief Get current robot pose estimate
     * @return 3D vector [x, y, theta]
     */
    Eigen::Vector3d getPose() const;

    /**
     * @brief Get current pose covariance (3x3 block)
     * @return 3x3 covariance matrix for [x, y, theta]
     */
    Eigen::Matrix3d getPoseCovariance() const;

    /**
     * @brief Get landmark position
     * @param landmark_id Index of landmark
     * @return 2D vector [mx, my]
     */
    Eigen::Vector2d getLandmark(int landmark_id) const;

    /**
     * @brief Get number of landmarks in map
     * @return Number of landmarks
     */
    int getNumLandmarks() const { return n_landmarks_; }

    /**
     * @brief Get full state vector (for debugging/visualization)
     * @return State vector [x, y, theta, mx1, my1, ...]
     */
    const Eigen::VectorXd& getState() const { return mu_; }

    /**
     * @brief Get full covariance matrix (for debugging/visualization)
     * @return Covariance matrix
     */
    const Eigen::MatrixXd& getCovariance() const { return Sigma_; }

private:
    // State
    Eigen::VectorXd mu_;        // State vector [x, y, theta, mx1, my1, mx2, my2, ...]
    Eigen::MatrixXd Sigma_;     // Full covariance matrix
    int n_landmarks_;           // Current number of landmarks

    // Noise parameters
    Eigen::Matrix3d R_;         // Process noise covariance (motion model)
    Eigen::Matrix2d Q_;         // Measurement noise covariance (sensor model)
    double mahalanobis_threshold_;  // Chi-squared threshold for data association

    /**
     * @brief Compute motion model Jacobian G
     * @param vx Linear velocity x (m/s)
     * @param vy Linear velocity y (m/s)
     * @param omega Angular velocity (rad/s)
     * @param dt Time step (s)
     * @return 3x3 Jacobian matrix
     */
    Eigen::Matrix3d computeG(double vx, double vy, double omega, double dt);

    /**
     * @brief Compute predicted measurement for a landmark
     * @param landmark_id Index of landmark
     * @return 2D vector [rho_expected, theta_expected]
     */
    Eigen::Vector2d computeH(int landmark_id) const;

    /**
     * @brief Compute measurement model Jacobian H
     * @param landmark_id Index of landmark
     * @return Jacobian matrix (2 x state_dim)
     */
    Eigen::MatrixXd computeHJacobian(int landmark_id) const;

    /**
     * @brief Normalize angle to [-pi, pi]
     * @param angle Input angle (rad)
     * @return Normalized angle
     */
    double normalizeAngle(double angle) const;
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_EKF_CORE_HPP
