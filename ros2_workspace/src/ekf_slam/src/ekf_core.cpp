#include "ekf_slam/ekf_core.hpp"
#include <cmath>

namespace ekf_slam {

EKFCore::EKFCore(const Eigen::Matrix4d& process_noise,
                 const Eigen::Matrix3d& scanmatch_noise)
    : R_(process_noise),
      Q_scanmatch_(scanmatch_noise) {
    mu_    = Eigen::VectorXd::Zero(4);
    Sigma_ = Eigen::MatrixXd::Identity(4, 4) * 0.1;
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

    // Landmarks are static; only the pose block has dynamics.
    Eigen::Matrix4d G_pose = computePoseJacobian(vx, vy, dt);

    const int N = stateDim();
    if (N == 4) {
        Sigma_ = G_pose * Sigma_ * G_pose.transpose() + R_;
        return;
    }

    // Pose-pose block
    Eigen::Matrix4d Sigma_pp = Sigma_.topLeftCorner<4, 4>();
    Sigma_.topLeftCorner<4, 4>() = G_pose * Sigma_pp * G_pose.transpose() + R_;

    // Pose-landmark cross blocks: G * Sigma_pl on the right, Sigma_lp * G^T on the left.
    if (N > 4) {
        Eigen::MatrixXd Sigma_pl = Sigma_.topRightCorner(4, N - 4);
        Sigma_.topRightCorner(4, N - 4)    = G_pose * Sigma_pl;
        Sigma_.bottomLeftCorner(N - 4, 4)  = Sigma_.topRightCorner(4, N - 4).transpose();
    }
    // Landmark-landmark block is left untouched (static landmarks, no process noise).
}

Eigen::Matrix4d EKFCore::computePoseJacobian(double vx, double vy, double dt) const {
    double theta = mu_(3);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

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

    const int N = stateDim();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, N);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(2, 3) = 1.0;

    Eigen::Matrix3d S   = H * Sigma_ * H.transpose() + Q;
    Eigen::MatrixXd K   = Sigma_ * H.transpose() * S.inverse();  // N x 3

    Eigen::Vector3d nu;
    nu(0) = inno_x;
    nu(1) = inno_y;
    nu(2) = normalizeAngle(dtheta - mu_(3));

    mu_    = mu_ + K * nu;
    mu_(3) = normalizeAngle(mu_(3));

    Sigma_ = (Eigen::MatrixXd::Identity(N, N) - K * H) * Sigma_;
}

void EKFCore::updateYaw(double yaw_meas, double yaw_noise) {
    double nu = normalizeAngle(yaw_meas - mu_(3));
    double S  = Sigma_(3, 3) + yaw_noise;
    if (S <= 0.0) return;

    // Expanded gain: full N-vector picking out column 3, scaled by 1/S.
    Eigen::VectorXd K = Sigma_.col(3) / S;

    mu_   += K * nu;
    mu_(3) = normalizeAngle(mu_(3));

    Sigma_ -= K * Sigma_.row(3);
}

void EKFCore::updateZ(double z_meas, double z_noise) {
    double nu = z_meas - mu_(2);

    double threshold;
    if (mu_(2) < 0.1) {
        threshold = 0.5;
    } else if (std::abs(commanded_vz_) > 0.05) {
        threshold = 0.4;
    } else {
        threshold = 0.1;
    }

    if (std::abs(nu) > threshold) return;

    double S  = Sigma_(2, 2) + z_noise;
    if (S <= 0.0) return;

    Eigen::VectorXd K = Sigma_.col(2) / S;
    mu_ += K * nu;
    Sigma_ -= K * Sigma_.row(2);
}

void EKFCore::setCommandedVz(double vz) {
    commanded_vz_ = vz;
}

int EKFCore::augment(const Eigen::VectorXd& new_state_block,
                     const Eigen::MatrixXd& new_block_cov,
                     const Eigen::MatrixXd& cross_cov) {
    const int N = stateDim();
    const int k = static_cast<int>(new_state_block.size());
    const int M = N + k;

    Eigen::VectorXd new_mu = Eigen::VectorXd::Zero(M);
    new_mu.head(N) = mu_;
    new_mu.tail(k) = new_state_block;

    Eigen::MatrixXd new_Sigma = Eigen::MatrixXd::Zero(M, M);
    new_Sigma.topLeftCorner(N, N)     = Sigma_;
    new_Sigma.bottomRightCorner(k, k) = new_block_cov;
    // cross_cov is (k x N): block ↔ existing state correlation.
    new_Sigma.bottomLeftCorner(k, N)  = cross_cov;
    new_Sigma.topRightCorner(N, k)    = cross_cov.transpose();

    mu_    = std::move(new_mu);
    Sigma_ = std::move(new_Sigma);
    return N;  // index where the new block starts
}

Eigen::Vector4d EKFCore::getPose() const {
    return mu_.head<4>();
}

Eigen::Matrix4d EKFCore::getPoseCovariance() const {
    return Sigma_.topLeftCorner<4, 4>();
}

double EKFCore::normalizeAngle(double angle) {
    while (angle >  M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

}  // namespace ekf_slam
