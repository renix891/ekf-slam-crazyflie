#ifndef CRAZYFLIE_NAVIGATION__NAVIGATION_NODE_HPP_
#define CRAZYFLIE_NAVIGATION__NAVIGATION_NODE_HPP_

#include <optional>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace crazyflie_navigation
{

enum class WaypointState
{
  MOVING,
  SCANNING,
  LANDING,
  IDLE
};

class NavigationNode : public rclcpp::Node
{
public:
  NavigationNode();

private:
  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void path_callback(const nav_msgs::msg::Path::SharedPtr msg);
  void enable_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);
  void control_loop();
  void status_loop();

  bool rotate_to_target(double target_yaw);
  void publish_zero_velocity();
  double altitude_hold_vz() const;

  static double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q);
  static double normalize_angle(double angle);
  static void world_to_body_frame(
    double vx_world, double vy_world, double yaw,
    double & vx_body, double & vy_body);

  // Parameters
  double waypoint_tolerance_;
  double scanning_yaw_rate_;
  double scan_timeout_s_;
  double flight_height_;
  double rotation_tolerance_;
  double yaw_kp_;
  double vz_kp_;
  double vz_max_;
  double max_velocity_;
  double approach_distance_;
  double approach_velocity_;

  // State
  bool enabled_;
  geometry_msgs::msg::PoseStamped::SharedPtr current_pose_;
  std::vector<geometry_msgs::msg::PoseStamped> planned_path_;
  size_t current_waypoint_idx_;
  WaypointState waypoint_state_;
  std::optional<double> target_yaw_;
  std::optional<rclcpp::Time> scan_started_at_;

  // IO
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::TimerBase::SharedPtr status_timer_;
};

}  // namespace crazyflie_navigation

#endif  // CRAZYFLIE_NAVIGATION__NAVIGATION_NODE_HPP_
