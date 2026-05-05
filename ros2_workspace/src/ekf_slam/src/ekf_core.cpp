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

int EKFCore::augmentLineFromObservation(double rho_obs_r, double theta_obs_r,
                                        const Eigen::Matrix2d& R_line) {
    // Inverse model: (rho_w, theta_w) from robot-frame observation.
    const double x   = mu_(0);
    const double y   = mu_(1);
    const double psi = mu_(3);

    double theta_w = normalizeAngle(theta_obs_r + psi);
    double rho_w   = rho_obs_r + x * std::cos(theta_w) + y * std::sin(theta_w);

    // Canonicalize: rho_w >= 0, normal points away from origin.
    if (rho_w < 0.0) {
        rho_w   = -rho_w;
        theta_w = normalizeAngle(theta_w + M_PI);
    }

    const int N = stateDim();
    Eigen::VectorXd new_block(2);
    new_block << rho_w, theta_w;

    // J_pose = d(rho_w, theta_w) / d(x, y, z, psi). Only x, y, psi columns nonzero.
    Eigen::MatrixXd J_pose = Eigen::MatrixXd::Zero(2, 4);
    J_pose(0, 0) = std::cos(theta_w);
    J_pose(0, 1) = std::sin(theta_w);
    J_pose(0, 3) = -x * std::sin(theta_w) + y * std::cos(theta_w);
    J_pose(1, 3) = 1.0;

    // J_obs = d(rho_w, theta_w) / d(rho_obs_r, theta_obs_r).
    Eigen::Matrix2d J_obs = Eigen::Matrix2d::Identity();
    J_obs(0, 1) = -x * std::sin(theta_w) + y * std::cos(theta_w);

    // New block covariance: pose-induced uncertainty + obs-induced uncertainty.
    Eigen::Matrix4d Sigma_pose = Sigma_.topLeftCorner<4, 4>();
    Eigen::Matrix2d new_block_cov =
        J_pose * Sigma_pose * J_pose.transpose() +
        J_obs  * R_line     * J_obs.transpose();

    // Cross-covariance with the existing state: derivative w.r.t. each existing
    // state column. Only the pose columns are non-zero (landmarks come in
    // through the existing pose-landmark cross-covariances).
    //   d new_block / d existing = J_pose * (rows 0..3 of Sigma_)
    Eigen::MatrixXd cross_cov = Eigen::MatrixXd::Zero(2, N);
    cross_cov = J_pose * Sigma_.topRows<4>();

    return augment(new_block, new_block_cov, cross_cov);
}

void EKFCore::predictLineObservation(int landmark_idx,
                                     Eigen::Vector2d& z_pred,
                                     Eigen::MatrixXd& H) const {
    const int N    = stateDim();
    const double x   = mu_(0);
    const double y   = mu_(1);
    const double psi = mu_(3);
    const double rho_j   = mu_(landmark_idx);
    const double theta_j = mu_(landmark_idx + 1);

    z_pred(0) = rho_j - x * std::cos(theta_j) - y * std::sin(theta_j);
    z_pred(1) = normalizeAngle(theta_j - psi);

    H = Eigen::MatrixXd::Zero(2, N);
    // Pose columns
    H(0, 0) = -std::cos(theta_j);
    H(0, 1) = -std::sin(theta_j);
    H(1, 3) = -1.0;
    // Landmark columns
    H(0, landmark_idx)     = 1.0;
    H(0, landmark_idx + 1) = x * std::sin(theta_j) - y * std::cos(theta_j);
    H(1, landmark_idx + 1) = 1.0;
}

void EKFCore::updateLineLandmark(int landmark_idx,
                                 double rho_obs_r, double theta_obs_r,
                                 const Eigen::Matrix2d& R_line) {
    const int N = stateDim();
    Eigen::Vector2d z_pred;
    Eigen::MatrixXd H;
    predictLineObservation(landmark_idx, z_pred, H);

    Eigen::Vector2d nu;
    nu(0) = rho_obs_r - z_pred(0);
    nu(1) = normalizeAngle(theta_obs_r - z_pred(1));

    Eigen::Matrix2d S = H * Sigma_ * H.transpose() + R_line;
    Eigen::MatrixXd K = Sigma_ * H.transpose() * S.inverse();  // N x 2

    mu_ += K * nu;
    mu_(3) = normalizeAngle(mu_(3));
    // Re-canonicalize each line landmark's theta.
    for (int j = 4; j + 1 < N; j += 2) {
        // Heuristic: lines come in pairs only if we ever interleave with corners,
        // but Stage 3 only adds line blocks. Treat every (rho, theta) pair as a
        // line and wrap theta. Stage 4 will switch on layout.
        mu_(j + 1) = normalizeAngle(mu_(j + 1));
    }

    Sigma_ = (Eigen::MatrixXd::Identity(N, N) - K * H) * Sigma_;
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
