#include "crazyflie_navigation/navigation_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace crazyflie_navigation
{

NavigationNode::NavigationNode()
: rclcpp::Node("autonomous_navigation_node"),
  enabled_(false),
  current_pose_(nullptr),
  current_waypoint_idx_(0),
  waypoint_state_(WaypointState::MOVING),
  target_yaw_(std::nullopt)
{
  // Parameters
  waypoint_tolerance_ = this->declare_parameter<double>("waypoint_tolerance", 0.2);
  scanning_yaw_rate_ = this->declare_parameter<double>("scanning_yaw_rate", 1.0);
  flight_height_ = this->declare_parameter<double>("flight_height", 0.3);
  rotation_tolerance_ = this->declare_parameter<double>("rotation_tolerance", 0.1);
  yaw_kp_ = this->declare_parameter<double>("yaw_kp", 2.0);
  vz_kp_ = this->declare_parameter<double>("vz_kp", 1.0);

  // Subscribers — use EKF-corrected pose, not raw odometry
  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/ekf_pose", 10,
    std::bind(&NavigationNode::pose_callback, this, std::placeholders::_1));

  path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
    "/planned_path", 10,
    std::bind(&NavigationNode::path_callback, this, std::placeholders::_1));

  // Publishers
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

  // Services
  enable_srv_ = this->create_service<std_srvs::srv::SetBool>(
    "/enable_autonomous",
    std::bind(&NavigationNode::enable_callback, this,
              std::placeholders::_1, std::placeholders::_2));

  // Timers
  control_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&NavigationNode::control_loop, this));
  status_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&NavigationNode::status_loop, this));

  RCLCPP_INFO(this->get_logger(), "Autonomous Navigation Node initialized (disabled)");
}

void NavigationNode::pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  bool was_none = (current_pose_ == nullptr);
  current_pose_ = msg;
  if (was_none) {
    RCLCPP_DEBUG(this->get_logger(),
      "Received first pose: (%.2f, %.2f, %.2f)",
      msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  }
}

void NavigationNode::path_callback(const nav_msgs::msg::Path::SharedPtr msg)
{
  if (!msg->poses.empty()) {
    planned_path_ = msg->poses;
    current_waypoint_idx_ = 0;
    RCLCPP_DEBUG(this->get_logger(),
      "Received path with %zu waypoints", msg->poses.size());
  }
}

void NavigationNode::enable_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  enabled_ = request->data;
  response->success = true;
  response->message = std::string("Autonomous navigation ") +
                      (enabled_ ? "enabled" : "disabled");
  RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
  if (!enabled_) {
    publish_zero_velocity();
    current_waypoint_idx_ = 0;
    waypoint_state_ = WaypointState::MOVING;
    target_yaw_ = std::nullopt;
  }
}

double NavigationNode::quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double NavigationNode::normalize_angle(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

void NavigationNode::world_to_body_frame(
  double vx_world, double vy_world, double yaw,
  double & vx_body, double & vy_body)
{
  // Rotation by -yaw
  double c = std::cos(-yaw);
  double s = std::sin(-yaw);
  vx_body = c * vx_world - s * vy_world;
  vy_body = s * vx_world + c * vy_world;
}

bool NavigationNode::rotate_to_target(double target_yaw)
{
  if (current_pose_ == nullptr) {
    return false;
  }

  double current_yaw = quaternion_to_yaw(current_pose_->pose.orientation);
  double yaw_error = normalize_angle(target_yaw - current_yaw);

  if (std::abs(yaw_error) < rotation_tolerance_) {
    RCLCPP_INFO(this->get_logger(),
      "Rotation complete - error=%.1f deg", yaw_error * 180.0 / M_PI);
    publish_zero_velocity();
    return true;
  }

  double yaw_rate_cmd = yaw_kp_ * yaw_error;
  yaw_rate_cmd = std::max(std::min(yaw_rate_cmd, scanning_yaw_rate_), -scanning_yaw_rate_);

  geometry_msgs::msg::Twist cmd;
  cmd.angular.z = yaw_rate_cmd;

  double z_err = flight_height_ - current_pose_->pose.position.z;
  cmd.linear.z = vz_kp_ * z_err;

  cmd_vel_pub_->publish(cmd);
  return false;
}

void NavigationNode::control_loop()
{
  if (!enabled_ || current_pose_ == nullptr) {
    return;
  }

  if (planned_path_.empty()) {
    publish_zero_velocity();
    return;
  }

  if (current_waypoint_idx_ >= planned_path_.size()) {
    publish_zero_velocity();
    return;
  }

  if (waypoint_state_ == WaypointState::MOVING) {
    const auto & waypoint = planned_path_[current_waypoint_idx_];
    double dx = waypoint.pose.position.x - current_pose_->pose.position.x;
    double dy = waypoint.pose.position.y - current_pose_->pose.position.y;
    double dist = std::hypot(dx, dy);

    bool is_final_waypoint = (current_waypoint_idx_ == planned_path_.size() - 1);
    double tolerance = is_final_waypoint ? (waypoint_tolerance_ / 5.0) : waypoint_tolerance_;

    if (dist < tolerance) {
      double current_yaw = quaternion_to_yaw(current_pose_->pose.orientation);
      target_yaw_ = normalize_angle(current_yaw + M_PI / 2.0);
      waypoint_state_ = WaypointState::SCANNING;
      RCLCPP_INFO(this->get_logger(),
        "Waypoint %zu reached - starting scan rotation to %.1f deg",
        current_waypoint_idx_, target_yaw_.value() * 180.0 / M_PI);
      return;
    }

    // Unit direction in world frame
    double inv_mag = 1.0 / std::max(dist, 1e-6);
    double vx_world = dx * inv_mag;
    double vy_world = dy * inv_mag;

    double current_yaw = quaternion_to_yaw(current_pose_->pose.orientation);
    double vx_body, vy_body;
    world_to_body_frame(vx_world, vy_world, current_yaw, vx_body, vy_body);

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = vx_body;
    cmd.linear.y = vy_body;

    double z_err = flight_height_ - current_pose_->pose.position.z;
    cmd.linear.z = vz_kp_ * z_err;

    cmd_vel_pub_->publish(cmd);

  } else if (waypoint_state_ == WaypointState::SCANNING) {
    if (target_yaw_.has_value() && rotate_to_target(target_yaw_.value())) {
      current_waypoint_idx_ += 1;
      waypoint_state_ = WaypointState::MOVING;
      RCLCPP_INFO(this->get_logger(),
        "Scan rotation complete - advancing to waypoint %zu", current_waypoint_idx_);
    }
  }
}

void NavigationNode::status_loop()
{
  if (!enabled_) {
    return;
  }

  std::ostringstream s;
  s << "AUTONOMOUS_NAV";

  if (current_pose_ == nullptr) {
    s << " | NO_POSE";
    RCLCPP_INFO(this->get_logger(), "Status: %s", s.str().c_str());
    return;
  }

  double curr_x = current_pose_->pose.position.x;
  double curr_y = current_pose_->pose.position.y;
  double curr_z = current_pose_->pose.position.z;
  double curr_yaw = quaternion_to_yaw(current_pose_->pose.orientation);

  char buf[256];
  std::snprintf(buf, sizeof(buf), " | pos=(%.2f,%.2f,%.2f) | yaw=%.1f deg",
                curr_x, curr_y, curr_z, curr_yaw * 180.0 / M_PI);
  s << buf;

  s << " | state=" << (waypoint_state_ == WaypointState::MOVING ? "MOVING" : "SCANNING");

  if (planned_path_.empty()) {
    s << " | NO_PATH";
  } else {
    std::snprintf(buf, sizeof(buf), " | waypoint=%zu/%zu",
                  current_waypoint_idx_, planned_path_.size());
    s << buf;

    if (current_waypoint_idx_ < planned_path_.size()) {
      if (waypoint_state_ == WaypointState::MOVING) {
        const auto & wp = planned_path_[current_waypoint_idx_];
        double dx = wp.pose.position.x - curr_x;
        double dy = wp.pose.position.y - curr_y;
        double dist = std::hypot(dx, dy);
        bool is_final = (current_waypoint_idx_ == planned_path_.size() - 1);
        double tol = is_final ? (waypoint_tolerance_ / 5.0) : waypoint_tolerance_;
        std::snprintf(buf, sizeof(buf), " | target=(%.2f,%.2f) | dist=%.2fm",
                      wp.pose.position.x, wp.pose.position.y, dist);
        s << buf;
        if (is_final) {
          std::snprintf(buf, sizeof(buf), " | FINAL_WP(tol=%.3fm)", tol);
          s << buf;
        }
      } else if (waypoint_state_ == WaypointState::SCANNING && target_yaw_.has_value()) {
        double yaw_error = normalize_angle(target_yaw_.value() - curr_yaw);
        std::snprintf(buf, sizeof(buf),
                      " | target_yaw=%.1f deg | yaw_error=%.1f deg",
                      target_yaw_.value() * 180.0 / M_PI,
                      yaw_error * 180.0 / M_PI);
        s << buf;
      }
    } else {
      s << " | PATH_COMPLETE";
    }
  }

  RCLCPP_INFO(this->get_logger(), "Status: %s", s.str().c_str());
}

void NavigationNode::publish_zero_velocity()
{
  cmd_vel_pub_->publish(geometry_msgs::msg::Twist());
}

}  // namespace crazyflie_navigation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<crazyflie_navigation::NavigationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
