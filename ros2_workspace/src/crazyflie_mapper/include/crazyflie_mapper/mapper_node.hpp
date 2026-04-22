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
#include <tf2_ros/static_transform_broadcaster.h>

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
  void apply_ray_weights(const std::vector<Cell> & ray_cells, int obstacle_distance_cells);

  // Parameters
  double avoidance_distance_;
  int max_avoidance_weight_;
  double max_obstacle_range_;
  double map_size_x_;
  double map_size_y_;
  double map_origin_x_;
  double map_origin_y_;
  double map_resolution_;

  // Derived map dimensions
  int map_width_;
  int map_height_;

  // State
  std::array<double, 3> position_{{0.0, 0.0, 0.0}};
  std::array<double, 3> angles_{{0.0, 0.0, 0.0}};
  std::vector<float> ranges_{0.0f, 0.0f, 0.0f, 0.0f};
  double range_max_{3.5};
  bool position_update_{false};

  std::vector<int8_t> map_;

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
};

}  // namespace crazyflie_mapper

#endif  // CRAZYFLIE_MAPPER__MAPPER_NODE_HPP_
