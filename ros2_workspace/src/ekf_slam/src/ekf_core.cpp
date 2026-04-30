#include "ekf_slam/ekf_core.hpp"
#include <cmath>

namespace ekf_slam {

EKFCore::EKFCore(const Eigen::Matrix4d& process_noise,
                 const Eigen::Matrix3d& scanmatch_noise)
    : R_(process_noise),
      Q_scanmatch_(scanmatch_noise) {
    mu_    = Eigen::Vector4d::Zero();
    Sigma_ = Eigen::Matrix4d::Identity() * 0.1;
}

void EKFCore::predict(double vx, double vy, double vz, double omega, double dt) {
    double theta = mu_(3);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double dx     = (vx * cos_theta - vy * sin_theta) * dt;
    double dy     = (vx * sin_theta + vy * cos_theta) * dt;
    double dz     = vz * dt;
    double dtheta = omega * dt;

    mu_(0) += dx;
    mu_(1) += dy;
    mu_(2) += dz;
    mu_(3)  = normalizeAngle(mu_(3) + dtheta);

    Eigen::Matrix4d G = computeG(vx, vy, omega, dt);
    Sigma_ = G * Sigma_ * G.transpose() + R_;
}

Eigen::Matrix4d EKFCore::computeG(double vx, double vy, double /*omega*/, double dt) {
    double theta = mu_(3);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    // Only x and y rows pick up off-diagonals from rotating the body-frame
    // velocity by theta.  z is independent of theta because vz is already
    // world-frame; theta is independent of all positions.
    Eigen::Matrix4d G = Eigen::Matrix4d::Identity();
    G(0, 3) = (-vx * sin_theta - vy * cos_theta) * dt;
    G(1, 3) = ( vx * cos_theta - vy * sin_theta) * dt;
    return G;
}

void EKFCore::updateScanMatch(double dx, double dy, double dtheta, double match_quality) {
    if (match_quality < 0.3) {
        return;
    }

    double inno_x = dx - mu_(0);
    double inno_y = dy - mu_(1);
    if (std::hypot(inno_x, inno_y) > 0.5) {
        return;
    }

    Eigen::Matrix3d Q = Q_scanmatch_ / (match_quality + 1e-6);

    // 3x4 H: rows pick state indices 0 (x), 1 (y), 3 (theta).
    Eigen::Matrix<double, 3, 4> H = Eigen::Matrix<double, 3, 4>::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 3) = 1.0;

    Eigen::Matrix3d S = H * Sigma_ * H.transpose() + Q;
    Eigen::Matrix<double, 4, 3> K = Sigma_ * H.transpose() * S.inverse();

    Eigen::Vector3d nu;
    nu(0) = inno_x;
    nu(1) = inno_y;
    nu(2) = normalizeAngle(dtheta - mu_(3));

    mu_    = mu_ + K * nu;
    mu_(3) = normalizeAngle(mu_(3));

    Sigma_ = (Eigen::Matrix4d::Identity() - K * H) * Sigma_;
}

void EKFCore::updateYaw(double yaw_meas, double yaw_noise) {
    double nu = normalizeAngle(yaw_meas - mu_(3));
    double S  = Sigma_(3, 3) + yaw_noise;
    if (S <= 0.0) return;

    Eigen::Vector4d K = Sigma_.col(3) / S;

    mu_   += K * nu;
    mu_(3) = normalizeAngle(mu_(3));

    Sigma_ -= K * Sigma_.row(3);
}

void EKFCore::updateZ(double z_meas, double z_noise) {
    double nu = z_meas - mu_(2);

    // Velocity-aware outlier gate. A downward-range innovation that's large
    // is only a true outlier when the drone is *not* deliberately changing
    // altitude — otherwise we'd reject takeoff/landing and freeze z.
    double threshold;
    if (mu_(2) < 0.1) {
        // Near the ground (initialization or just after touchdown) — let
        // any plausible reading in so z latches onto reality.
        threshold = 0.5;
    } else if (std::abs(commanded_vz_) > 0.05) {
        // Active takeoff or landing: real innovations can be large.
        threshold = 0.4;
    } else {
        // Hover: reject spikes that almost certainly came from the down-beam
        // hitting an obstacle below the drone.
        threshold = 0.1;
    }

    if (std::abs(nu) > threshold) return;

    double S  = Sigma_(2, 2) + z_noise;
    if (S <= 0.0) return;

    Eigen::Vector4d K = Sigma_.col(2) / S;

    mu_ += K * nu;

    Sigma_ -= K * Sigma_.row(2);
}

void EKFCore::setCommandedVz(double vz) {
    commanded_vz_ = vz;
}

Eigen::Vector4d EKFCore::getPose() const           { return mu_; }
Eigen::Matrix4d EKFCore::getPoseCovariance() const { return Sigma_; }

double EKFCore::normalizeAngle(double angle) const {
    while (angle >  M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

}  // namespace ekf_slam
