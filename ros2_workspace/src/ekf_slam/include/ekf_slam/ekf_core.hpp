#ifndef EKF_SLAM_EKF_CORE_HPP
#define EKF_SLAM_EKF_CORE_HPP

#include <Eigen/Dense>

namespace ekf_slam {

/**
 * @brief EKF localization with scan-matching correction.
 *
 * State vector: [x, y, theta]  (robot pose in world frame, 3x1)
 *
 * Motion model (predict): constant body-frame velocity.
 * Correction (updateScanMatch): SVD scan match between consecutive scans
 *   produces an absolute world-frame pose observation; H = I_3.
 */
class EKFCore {
public:
    EKFCore(const Eigen::Matrix3d& process_noise   = Eigen::Matrix3d::Identity() * 0.01,
            const Eigen::Matrix3d& scanmatch_noise = Eigen::Matrix3d::Identity() * 0.5);

    void predict(double vx, double vy, double omega, double dt);

    /**
     * @brief EKF correction from scan matching.
     *
     * (dx, dy, dtheta) is interpreted as an *absolute* world-frame pose
     * observation (the caller composes the relative scan-match transform
     * with the pose at the time of the previous scan).
     *
     * @param match_quality scalar in (0, 1]; noise is scaled as Q/match_quality.
     */
    void updateScanMatch(double dx, double dy, double dtheta, double match_quality);

    Eigen::Vector3d  getPose() const;
    Eigen::Matrix3d  getPoseCovariance() const;
    const Eigen::Vector3d& getState() const      { return mu_; }
    const Eigen::Matrix3d& getCovariance() const { return Sigma_; }

    // Previous scan storage (managed by the node)
    Eigen::Matrix2Xd previousScan_;
    bool             hasPreviousScan_ = false;

private:
    Eigen::Vector3d mu_;
    Eigen::Matrix3d Sigma_;

    Eigen::Matrix3d R_;             // process noise
    Eigen::Matrix3d Q_scanmatch_;   // scan-match measurement noise (base)

    Eigen::Matrix3d computeG(double vx, double vy, double omega, double dt);
    double normalizeAngle(double angle) const;
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_EKF_CORE_HPP
