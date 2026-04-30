#ifndef EKF_SLAM_EKF_CORE_HPP
#define EKF_SLAM_EKF_CORE_HPP

#include <Eigen/Dense>

namespace ekf_slam {

/**
 * @brief EKF localization with scan-matching and altitude correction.
 *
 * State vector: [x, y, z, theta]  (robot pose in world frame, 4x1)
 *   index 0 = x      (m, world)
 *   index 1 = y      (m, world)
 *   index 2 = z      (m, world; AGL on real flow-deck flights)
 *   index 3 = theta  (rad, yaw)
 *
 * Motion model (predict): x, y use the existing body-frame planar model
 *   rotated by theta; z integrates world-frame vz directly; theta integrates
 *   omega.  z and theta are independent in the predict-step Jacobian.
 *
 * Corrections:
 *   - updateScanMatch: scan-to-map alignment yields an absolute world-frame
 *     (x, y) observation; theta passthrough comes from updateYaw, since the
 *     4-beam multiranger underconstrains rotation.  H is 3x4 picking rows
 *     0, 1, 3.
 *   - updateYaw: 1-D Kalman correction on theta from odom yaw.
 *   - updateZ:   1-D Kalman correction on z from a downward range source
 *     (Gazebo gpu_lidar / real CF flow-deck VL53L1x).  H = e_z.
 */
class EKFCore {
public:
    EKFCore(const Eigen::Matrix4d& process_noise   = Eigen::Matrix4d::Identity() * 0.01,
            const Eigen::Matrix3d& scanmatch_noise = Eigen::Matrix3d::Identity() * 0.5);

    /**
     * @brief Predict step.
     * @param vx body-frame x velocity (m/s)
     * @param vy body-frame y velocity (m/s)
     * @param vz world-frame z velocity (m/s) — matches both Gazebo
     *           OdometryPublisher and the real CF flow-deck output for a
     *           level-hovering quad.
     * @param omega yaw rate (rad/s)
     * @param dt seconds since previous predict
     */
    void predict(double vx, double vy, double vz, double omega, double dt);

    /**
     * @brief EKF correction from scan matching.
     *
     * (dx, dy, dtheta) is interpreted as an absolute world-frame pose
     * observation.  z is unobserved.
     *
     * @param match_quality scalar in (0, 1]; noise is scaled as Q/match_quality.
     */
    void updateScanMatch(double dx, double dy, double dtheta, double match_quality);

    /**
     * @brief Direct 1-D Kalman correction on theta from an external yaw source
     *        (e.g. odometry quaternion).  H = e_theta.
     */
    void updateYaw(double yaw_meas, double yaw_noise = 0.01);

    /**
     * @brief Direct 1-D Kalman correction on z from a downward range source.
     *        H = e_z.  No angle wrapping.  On real flight, z_meas comes from
     *        the VL53L1x via /crazyflie/range/down — independent of the
     *        firmware's pose.z, so the innovation is real.
     */
    void updateZ(double z_meas, double z_noise = 0.01);

    /**
     * @brief Inform the EKF of the most recent commanded world-frame vz so
     *        updateZ() can adapt its outlier gate. During takeoff/landing the
     *        true innovation is large by design; during hover any large
     *        innovation is almost certainly a bad range reading (e.g. the
     *        down-beam struck a box on the floor).
     */
    void setCommandedVz(double vz);

    Eigen::Vector4d  getPose() const;
    Eigen::Matrix4d  getPoseCovariance() const;
    const Eigen::Vector4d& getState() const      { return mu_; }
    const Eigen::Matrix4d& getCovariance() const { return Sigma_; }

    Eigen::Matrix2Xd previousScan_;
    bool             hasPreviousScan_ = false;

private:
    Eigen::Vector4d mu_;
    Eigen::Matrix4d Sigma_;

    Eigen::Matrix4d R_;             // process noise (4x4)
    Eigen::Matrix3d Q_scanmatch_;   // scan-match measurement noise over (x, y, theta)

    // Most recent commanded world-frame vz (m/s). Drives the updateZ outlier
    // gate so it tightens during hover and relaxes during takeoff/landing.
    double commanded_vz_ = 0.0;

    Eigen::Matrix4d computeG(double vx, double vy, double omega, double dt);
    double normalizeAngle(double angle) const;
};

}  // namespace ekf_slam

#endif  // EKF_SLAM_EKF_CORE_HPP
