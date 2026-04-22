#ifndef CRAZYFLIE_NAVIGATION__BOX_LANDING_NODE_HPP_
#define CRAZYFLIE_NAVIGATION__BOX_LANDING_NODE_HPP_

#include <array>
#include <deque>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>

namespace crazyflie_navigation
{

enum class LandingPhase
{
  IDLE,
  GRID_SEARCH,
  NAVIGATE,
  WAIT_COMPLETE,
  STOP
};

enum class RangeDir
{
  FRONT = 0,
  BACK = 1,
  LEFT = 2,
  RIGHT = 3,
  DOWN = 4,
  NUM_DIRS = 5
};

struct GridSegment
{
  double vx_unit;
  double vy_unit;
  double duration;
};

class BoxLandingNode : public rclcpp::Node
{
public:
  BoxLandingNode();

private:
  void pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void range_cb(const sensor_msgs::msg::Range::SharedPtr msg, RangeDir dir);
  void enable_cb(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
    std::shared_ptr<std_srvs::srv::SetBool::Response> response);

  void loop();

  void detect_edges();
  std::vector<GridSegment> generate_grid_sequence();
  void compute_and_navigate_to_centroid();
  void call_autonomous_nav(bool enable);
  void publish_status(const std::string & status);

  // Parameters
  bool enabled_;
  double flight_height_;
  double search_span_;
  double search_step_;
  double vz_max_;
  double min_hover_;
  int min_edge_points_;
  double goal_tolerance_;
  double box_landing_speed_;
  double search_scale_;
  double edge_delta_threshold_;  // delta in range between consecutive samples to flag an edge

  // State
  geometry_msgs::msg::PoseStamped::SharedPtr pose_;
  LandingPhase phase_;
  std::vector<std::pair<double, double>> edge_points_;
  std::optional<std::pair<double, double>> box_centroid_;
  std::optional<double> wait_start_time_;

  // Grid search state
  std::optional<std::pair<double, double>> grid_center_;
  std::vector<GridSegment> grid_sequence_;
  std::optional<double> grid_start_time_;

  // Range sensor state
  std::array<std::optional<double>, 5> latest_range_;   // indexed by RangeDir
  std::array<std::optional<double>, 5> previous_range_;

  // IO
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  std::array<rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr, 5> range_subs_;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;

  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr autonomous_nav_client_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace crazyflie_navigation

#endif  // CRAZYFLIE_NAVIGATION__BOX_LANDING_NODE_HPP_
