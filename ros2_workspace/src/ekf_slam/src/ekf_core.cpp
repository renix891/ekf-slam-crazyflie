#include "ekf_slam/ekf_core.hpp"
#include <cmath>
#include <limits>
#include <iostream>

namespace ekf_slam {

EKFCore::EKFCore(const Eigen::Matrix3d& process_noise,
                 const Eigen::Matrix2d& measurement_noise,
                 double mahalanobis_threshold)
    : n_landmarks_(0),
      R_(process_noise),
      Q_(measurement_noise),
      mahalanobis_threshold_(mahalanobis_threshold) {
    // Initialize state vector with robot pose only [x, y, theta]
    mu_ = Eigen::Vector3d::Zero();

    // Initialize covariance matrix (3x3 for pose only initially)
    Sigma_ = Eigen::Matrix3d::Identity() * 0.1;
}

void EKFCore::predict(double vx, double vy, double omega, double dt) {
    // Extract current pose
    double theta = mu_(2);

    // Motion model: update pose based on velocity commands (body frame)
    // Transform body-frame velocities to world frame
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double dx = (vx * cos_theta - vy * sin_theta) * dt;
    double dy = (vx * sin_theta + vy * cos_theta) * dt;
    double dtheta = omega * dt;

    // Update robot pose in state vector
    mu_(0) += dx;
    mu_(1) += dy;
    mu_(2) = normalizeAngle(mu_(2) + dtheta);

    // Landmarks don't move (static map assumption)
    // So landmark portions of mu_ remain unchanged

    // Compute Jacobian G (derivative of motion model w.r.t. pose)
    Eigen::Matrix3d G = computeG(vx, vy, omega, dt);

    // Build full state Jacobian F_x
    // F_x is identity except for top-left 3x3 block which is G
    int state_dim = mu_.size();
    Eigen::MatrixXd F_x = Eigen::MatrixXd::Identity(state_dim, state_dim);
    F_x.block<3, 3>(0, 0) = G;

    // Update covariance: � = F_x * � * F_x^T + F_u * R * F_u^T
    // F_u maps control noise to state (here it's just top-left 3x3 identity)
    Eigen::MatrixXd F_u = Eigen::MatrixXd::Zero(state_dim, 3);
    F_u.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    Sigma_ = F_x * Sigma_ * F_x.transpose() + F_u * R_ * F_u.transpose();
}

Eigen::Matrix3d EKFCore::computeG(double vx, double vy, double /*omega*/, double dt) {
    // Jacobian of motion model with respect to pose [x, y, theta]
    // g(x, u) = [x + (vx*cos(theta) - vy*sin(theta))*dt,
    //            y + (vx*sin(theta) + vy*cos(theta))*dt,
    //            theta + omega*dt]

    double theta = mu_(2);
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    Eigen::Matrix3d G = Eigen::Matrix3d::Identity();

    // g/x = 1, g/y = 1 (diagonal elements already set by Identity)

    // g_x/theta = (-vx*sin(theta) - vy*cos(theta))*dt
    G(0, 2) = (-vx * sin_theta - vy * cos_theta) * dt;

    // g_y/theta = (vx*cos(theta) - vy*sin(theta))*dt
    G(1, 2) = (vx * cos_theta - vy * sin_theta) * dt;

    // g_theta/theta = 1 (already set by Identity)

    return G;
}

void EKFCore::update(double rho, double theta_obs, int landmark_id) {
    if (landmark_id < 0 || landmark_id >= n_landmarks_) {
        std::cerr << "Invalid landmark_id: " << landmark_id << std::endl;
        return;
    }

    // Compute expected measurement
    Eigen::Vector2d z_expected = computeH(landmark_id);

    // Measurement residual (innovation)
    Eigen::Vector2d z_measured(rho, theta_obs);
    Eigen::Vector2d nu = z_measured - z_expected;

    // Normalize bearing angle difference to [-pi, pi]
    nu(1) = normalizeAngle(nu(1));

    // Compute measurement Jacobian H
    Eigen::MatrixXd H = computeHJacobian(landmark_id);

    // Innovation covariance: S = H * � * H^T + Q
    Eigen::Matrix2d S = H * Sigma_ * H.transpose() + Q_;

    // Kalman gain: K = � * H^T * S^{-1}
    Eigen::MatrixXd K = Sigma_ * H.transpose() * S.inverse();

    // State update: � = � + K * �
    mu_ = mu_ + K * nu;

    // Normalize robot orientation
    mu_(2) = normalizeAngle(mu_(2));

    // Covariance update: � = (I - K * H) * �
    int state_dim = mu_.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_dim, state_dim);
    Sigma_ = (I - K * H) * Sigma_;
}

int EKFCore::addLandmark(double rho, double theta_obs) {
    // Convert range-bearing measurement to Cartesian coordinates in world frame
    double x = mu_(0);
    double y = mu_(1);
    double theta = mu_(2);

    // Bearing in world frame
    double theta_world = theta + theta_obs;

    // Landmark position in world frame
    double mx = x + rho * std::cos(theta_world);
    double my = y + rho * std::sin(theta_world);

    // Extend state vector
    int old_dim = mu_.size();
    int new_dim = old_dim + 2;

    mu_.conservativeResize(new_dim);
    mu_(old_dim) = mx;
    mu_(old_dim + 1) = my;

    // Extend covariance matrix
    Sigma_.conservativeResize(new_dim, new_dim);

    // Initialize new landmark covariance (high uncertainty)
    Sigma_.block(old_dim, 0, 2, old_dim).setZero();
    Sigma_.block(0, old_dim, old_dim, 2).setZero();
    Sigma_(old_dim, old_dim) = 10.0;      // High initial uncertainty in x
    Sigma_(old_dim + 1, old_dim + 1) = 10.0;  // High initial uncertainty in y
    Sigma_(old_dim, old_dim + 1) = 0.0;
    Sigma_(old_dim + 1, old_dim) = 0.0;

    int landmark_id = n_landmarks_;
    n_landmarks_++;

    return landmark_id;
}

int EKFCore::associateLandmark(double rho, double theta_obs) {
    // If no landmarks exist, this is definitely a new one
    if (n_landmarks_ == 0) {
        return -1;
    }

    // Find closest landmark using Mahalanobis distance
    int best_landmark = -1;
    double min_mahalanobis = std::numeric_limits<double>::max();

    for (int i = 0; i < n_landmarks_; i++) {
        // Compute expected measurement for this landmark
        Eigen::Vector2d z_expected = computeH(i);

        // Innovation
        Eigen::Vector2d z_measured(rho, theta_obs);
        Eigen::Vector2d nu = z_measured - z_expected;
        nu(1) = normalizeAngle(nu(1));

        // Compute measurement Jacobian
        Eigen::MatrixXd H = computeHJacobian(i);

        // Innovation covariance
        Eigen::Matrix2d S = H * Sigma_ * H.transpose() + Q_;

        // Mahalanobis distance: d^2 = nu^T * S^{-1} * nu
        double mahalanobis_dist = nu.transpose() * S.inverse() * nu;

        std::cerr << "[assoc] landmark " << i
                  << " d^2=" << mahalanobis_dist
                  << " (thr=" << mahalanobis_threshold_ << ")" << std::endl;

        if (mahalanobis_dist < min_mahalanobis) {
            min_mahalanobis = mahalanobis_dist;
            best_landmark = i;
        }
    }

    // Check if best match is within threshold
    if (min_mahalanobis < mahalanobis_threshold_) {
        std::cerr << "[assoc] matched landmark " << best_landmark
                  << " d^2=" << min_mahalanobis << std::endl;
        return best_landmark;
    } else {
        std::cerr << "[assoc] no match, best d^2=" << min_mahalanobis
                  << " >= thr=" << mahalanobis_threshold_ << std::endl;
        return -1;  // No match found, new landmark
    }
}

Eigen::Vector2d EKFCore::computeH(int landmark_id) const {
    // Extract robot pose
    double x = mu_(0);
    double y = mu_(1);
    double theta = mu_(2);

    // Extract landmark position
    int landmark_idx = 3 + 2 * landmark_id;
    double mx = mu_(landmark_idx);
    double my = mu_(landmark_idx + 1);

    // Compute expected measurement
    double dx = mx - x;
    double dy = my - y;

    double rho = std::sqrt(dx * dx + dy * dy);
    double phi = std::atan2(dy, dx) - theta;
    phi = normalizeAngle(phi);

    return Eigen::Vector2d(rho, phi);
}

Eigen::MatrixXd EKFCore::computeHJacobian(int landmark_id) const {
    // Jacobian of measurement model h(x) with respect to full state
    // h = [rho, phi]^T where:
    //   rho = sqrt((mx - x)^2 + (my - y)^2)
    //   phi = atan2(my - y, mx - x) - theta

    double x = mu_(0);
    double y = mu_(1);

    int landmark_idx = 3 + 2 * landmark_id;
    double mx = mu_(landmark_idx);
    double my = mu_(landmark_idx + 1);

    double dx = mx - x;
    double dy = my - y;
    double q = dx * dx + dy * dy;
    double sqrt_q = std::sqrt(q);

    int state_dim = mu_.size();
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_dim);

    // Avoid division by zero
    if (sqrt_q < 1e-6) {
        return H;
    }

    // rho/x = -dx/sqrt(q)
    H(0, 0) = -dx / sqrt_q;

    // rho/y = -dy/sqrt(q)
    H(0, 1) = -dy / sqrt_q;

    // rho/theta = 0
    H(0, 2) = 0.0;

    // phi/x = dy/q
    H(1, 0) = dy / q;

    // phi/y = -dx/q
    H(1, 1) = -dx / q;

    // phi/theta = -1
    H(1, 2) = -1.0;

    // Derivatives with respect to landmark position
    // rho/mx = dx/sqrt(q)
    H(0, landmark_idx) = dx / sqrt_q;

    // rho/my = dy/sqrt(q)
    H(0, landmark_idx + 1) = dy / sqrt_q;

    // phi/mx = -dy/q
    H(1, landmark_idx) = -dy / q;

    // phi/my = dx/q
    H(1, landmark_idx + 1) = dx / q;

    return H;
}

Eigen::Vector3d EKFCore::getPose() const {
    return mu_.head<3>();
}

Eigen::Matrix3d EKFCore::getPoseCovariance() const {
    return Sigma_.block<3, 3>(0, 0);
}

Eigen::Vector2d EKFCore::getLandmark(int landmark_id) const {
    if (landmark_id < 0 || landmark_id >= n_landmarks_) {
        return Eigen::Vector2d::Zero();
    }

    int landmark_idx = 3 + 2 * landmark_id;
    return Eigen::Vector2d(mu_(landmark_idx), mu_(landmark_idx + 1));
}

double EKFCore::normalizeAngle(double angle) const {
    // Normalize angle to [-pi, pi]
    while (angle > M_PI) {
        angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
        angle += 2.0 * M_PI;
    }
    return angle;
}

}  // namespace ekf_slam
