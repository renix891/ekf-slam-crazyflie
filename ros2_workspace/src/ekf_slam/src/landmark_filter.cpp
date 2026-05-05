#include "ekf_slam/landmark_filter.hpp"
#include "ekf_slam/ekf_core.hpp"   // for EKFCore::normalizeAngle

#include <cmath>

namespace ekf_slam {

int LandmarkFilter::augmentLine(const Eigen::Vector4d& pose,
                                double rho_obs_r, double theta_obs_r,
                                const Eigen::Matrix2d& R_line) {
    // Inverse model — robot-frame observation → world-frame (rho_w, theta_w).
    // Storage convention matches the previous joint EKF: rho_w is signed,
    // predictLineObservation returns the matching signed prediction.
    const double x   = pose(0);
    const double y   = pose(1);
    const double psi = pose(3);

    const double theta_w = EKFCore::normalizeAngle(theta_obs_r + psi);
    const double rho_w   = rho_obs_r + x * std::cos(theta_w) +
                                       y * std::sin(theta_w);

    // J_obs = d(rho_w, theta_w) / d(rho_obs_r, theta_obs_r). Pose is treated
    // as a constant input so it has no Jacobian here.
    Eigen::Matrix2d J_obs = Eigen::Matrix2d::Identity();
    J_obs(0, 1) = -x * std::sin(theta_w) + y * std::cos(theta_w);

    Eigen::Matrix2d new_block_cov = J_obs * R_line * J_obs.transpose();

    const int N_old = stateDim();           // 2 * old N landmarks
    const int N_new = N_old + 2;

    // Resize state and covariance to add a new 2-vector block at the bottom.
    Eigen::VectorXd mu_new(N_new);
    if (N_old > 0) mu_new.head(N_old) = mu_lm_;
    mu_new(N_old)     = rho_w;
    mu_new(N_old + 1) = theta_w;

    Eigen::MatrixXd Sigma_new = Eigen::MatrixXd::Zero(N_new, N_new);
    if (N_old > 0) {
        Sigma_new.topLeftCorner(N_old, N_old) = Sigma_lm_;
    }
    Sigma_new.bottomRightCorner<2, 2>() = new_block_cov;
    // No cross-covariance with existing landmarks: a fresh observation has no
    // prior correlation to landmarks observed before it. Updates will build
    // up cross-covariance organically as both blocks are corrected together.

    mu_lm_    = mu_new;
    Sigma_lm_ = Sigma_new;

    return N_old / 2;  // index of the new landmark
}

void LandmarkFilter::predictLineObservation(const Eigen::Vector4d& pose,
                                            int landmark_idx,
                                            Eigen::Vector2d& z_pred,
                                            Eigen::Matrix2d& H_lm) const {
    const double x   = pose(0);
    const double y   = pose(1);
    const double psi = pose(3);

    const int row = 2 * landmark_idx;
    const double rho_j   = mu_lm_(row);
    const double theta_j = mu_lm_(row + 1);

    // Forward model: project landmark back into robot-frame measurement.
    z_pred(0) = rho_j - x * std::cos(theta_j) - y * std::sin(theta_j);
    z_pred(1) = EKFCore::normalizeAngle(theta_j - psi);

    // H_lm = d(z) / d(rho_j, theta_j). The pose Jacobian columns from the old
    // joint EKF are dropped — pose is a constant input.
    H_lm(0, 0) = 1.0;
    H_lm(0, 1) = x * std::sin(theta_j) - y * std::cos(theta_j);
    H_lm(1, 0) = 0.0;
    H_lm(1, 1) = 1.0;
}

bool LandmarkFilter::updateLine(const Eigen::Vector4d& pose,
                                int landmark_idx,
                                double rho_obs_r, double theta_obs_r,
                                const Eigen::Matrix2d& R_line) {
    const int N = stateDim();
    if (landmark_idx < 0 || 2 * landmark_idx + 1 >= N) return false;

    Eigen::Vector2d z_pred;
    Eigen::Matrix2d H_lm;
    predictLineObservation(pose, landmark_idx, z_pred, H_lm);

    // Build the full-state H (2 x 2N) so the gain K is full-state and the
    // landmark block's correction propagates to its cross-covariances with
    // every other landmark. Only the columns of the observed landmark are
    // nonzero — pose has no row at all in this filter.
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, N);
    const int col = 2 * landmark_idx;
    H.block<2, 2>(0, col) = H_lm;

    Eigen::Vector2d nu;
    nu(0) = rho_obs_r - z_pred(0);
    nu(1) = EKFCore::normalizeAngle(theta_obs_r - z_pred(1));

    Eigen::Matrix2d S = H * Sigma_lm_ * H.transpose() + R_line;

    // Conditioning gate (lifted from joint EKF): bail if S is near-singular,
    // otherwise S^-1 explodes and the gain swings the state by enormous
    // amounts.
    const double det_S  = S.determinant();
    const double norm_S = S.norm();
    if (!std::isfinite(det_S) || std::abs(det_S) < 1e-9 || norm_S < 1e-6) {
        return false;
    }

    Eigen::MatrixXd K = Sigma_lm_ * H.transpose() * S.inverse();   // N x 2

    mu_lm_ += K * nu;
    normalizeStateAngles();

    Sigma_lm_ = (Eigen::MatrixXd::Identity(N, N) - K * H) * Sigma_lm_;
    return true;
}

void LandmarkFilter::normalizeStateAngles() {
    // Every odd-indexed state entry is a theta.
    for (int i = 1; i < mu_lm_.size(); i += 2) {
        mu_lm_(i) = EKFCore::normalizeAngle(mu_lm_(i));
    }
}

}  // namespace ekf_slam
