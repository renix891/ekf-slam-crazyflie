#include "ekf_slam/ekf_core.hpp"
#include <cmath>

namespace ekf_slam {

EKFCore::EKFCore(const Eigen::Matrix3d& process_noise,
                 const Eigen::Matrix3d& scanmatch_noise)
    : R_(process_noise),
      Q_scanmatch_(scanmatch_noise) {
    mu_    = Eigen::Vector3d::Zero();
    Sigma_ = Eigen::Matrix3d::Identity() * 0.1;
}

void EKFCore::predict(double vx, double vy, double omega, double dt) {
    double theta = mu_(2);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double dx     = (vx * cos_theta - vy * sin_theta) * dt;
    double dy     = (vx * sin_theta + vy * cos_theta) * dt;
    double dtheta = omega * dt;

    mu_(0) += dx;
    mu_(1) += dy;
    mu_(2)  = normalizeAngle(mu_(2) + dtheta);

    Eigen::Matrix3d G = computeG(vx, vy, omega, dt);
    Sigma_ = G * Sigma_ * G.transpose() + R_;
}

Eigen::Matrix3d EKFCore::computeG(double vx, double vy, double /*omega*/, double dt) {
    double theta = mu_(2);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    Eigen::Matrix3d G = Eigen::Matrix3d::Identity();
    G(0, 2) = (-vx * sin_theta - vy * cos_theta) * dt;
    G(1, 2) = ( vx * cos_theta - vy * sin_theta) * dt;
    return G;
}

void EKFCore::updateScanMatch(double dx, double dy, double dtheta, double match_quality) {
    // Reject low-confidence scan matches outright; better to coast on the
    // predict step than absorb a bad correction.
    if (match_quality < 0.3) {
        return;
    }

    Eigen::Matrix3d Q = Q_scanmatch_ / (match_quality + 1e-6);

    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d S = H * Sigma_ * H.transpose() + Q;
    Eigen::Matrix3d K = Sigma_ * H.transpose() * S.inverse();

    Eigen::Vector3d nu;
    nu(0) = dx     - mu_(0);
    nu(1) = dy     - mu_(1);
    nu(2) = normalizeAngle(dtheta - mu_(2));

    mu_    = mu_ + K * nu;
    mu_(2) = normalizeAngle(mu_(2));

    Sigma_ = (Eigen::Matrix3d::Identity() - K * H) * Sigma_;
}

Eigen::Vector3d EKFCore::getPose() const           { return mu_; }
Eigen::Matrix3d EKFCore::getPoseCovariance() const { return Sigma_; }

double EKFCore::normalizeAngle(double angle) const {
    while (angle >  M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

}  // namespace ekf_slam
