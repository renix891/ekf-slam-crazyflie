#ifndef CRAZYFLIE_MAPPER__MAPPER_NODE_HPP_
#define CRAZYFLIE_MAPPER__MAPPER_NODE_HPP_

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

namespace crazyflie_mapper
{

class MapperNode : public rclcpp::Node
{
public:
  MapperNode();

private:
  using Cell = std::pair<int, int>;

  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void publish_map();

  std::vector<std::array<double, 3>> rotate_and_create_points() const;
  static std::array<double, 3> rotate(
    double roll, double pitch, double yaw,
    const std::array<double, 3> & origin,
    const std::array<double, 3> & point);
  static double yaw_from_quaternion(double x, double y, double z, double w);
  static void euler_from_quaternion(
    double x, double y, double z, double w,
    double & roll, double & pitch, double & yaw);
  static std::vector<Cell> bresenham_line(int x0, int y0, int x1, int y1);

  // Parameters
  double max_obstacle_range_;
  double map_size_x_;
  double map_size_y_;
  double map_origin_x_;
  double map_origin_y_;
  double map_resolution_;
  // Log-odds update parameters
  float l_free_;          // increment per free observation (negative)
  float l_occ_;           // increment per occupied observation (positive)
  float l_min_;           // saturation floor
  float l_max_;           // saturation ceiling
  float p_occ_thresh_;    // probability above which a cell is published as 100
  float p_free_thresh_;   // probability below which a cell is published as 0
  int dilation_cells_;    // C-space inflation radius in cells

  // Derived map dimensions
  int map_width_;
  int map_height_;

  // State
  std::array<double, 3> position_{{0.0, 0.0, 0.0}};
  std::array<double, 3> angles_{{0.0, 0.0, 0.0}};
  std::vector<float> ranges_{0.0f, 0.0f, 0.0f, 0.0f};
  double range_max_{3.5};
  bool position_update_{false};

  // Persistent log-odds grid — accumulates across all scans, never reset.
  std::vector<float> logodds_;

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

}  // namespace crazyflie_mapper

#endif  // CRAZYFLIE_MAPPER__MAPPER_NODE_HPP_
