#include "crazyflie_mapper/mapper_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#include <geometry_msgs/msg/transform_stamped.hpp>

namespace crazyflie_mapper
{

MapperNode::MapperNode()
: rclcpp::Node("crazyflie_mapper_node")
{
  // Parameters (defaults mirror the Python simple_mapper_node.py)
  avoidance_distance_    = this->declare_parameter<double>("avoidance_distance", 0.6);
  max_avoidance_weight_  = this->declare_parameter<int>("max_avoidance_weight", 50);
  max_obstacle_range_    = this->declare_parameter<double>("max_obstacle_range", 1.0);

  map_size_x_    = this->declare_parameter<double>("map_size_x", 40.0);
  map_size_y_    = this->declare_parameter<double>("map_size_y", 20.0);
  map_origin_x_  = this->declare_parameter<double>("map_origin_x", -10.0);
  map_origin_y_  = this->declare_parameter<double>("map_origin_y", -10.0);
  map_resolution_ = this->declare_parameter<double>("map_resolution", 0.1);

  map_width_  = static_cast<int>(map_size_x_ / map_resolution_);
  map_height_ = static_cast<int>(map_size_y_ / map_resolution_);
  map_.assign(static_cast<size_t>(map_width_) * static_cast<size_t>(map_height_),
              static_cast<int8_t>(-1));

  // Subscribers — EKF-corrected pose and multiranger scan
  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/ekf_pose", 10,
    std::bind(&MapperNode::pose_callback, this, std::placeholders::_1));

  scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
    "/crazyflie/scan", 10,
    std::bind(&MapperNode::scan_callback, this, std::placeholders::_1));

  // Transient-local latched publisher so late subscribers still receive the map
  rclcpp::QoS map_qos(rclcpp::KeepLast(1));
  map_qos.transient_local();
  map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", map_qos);

  publish_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&MapperNode::publish_map, this));

  // Static TF: map -> odom (identity). EKF corrects inside the odom frame,
  // so map and odom coincide. Publishing this lets RViz and the planner
  // resolve TF lookups between the two frames.
  static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(this);
  geometry_msgs::msg::TransformStamped t_map;
  t_map.header.stamp = this->get_clock()->now();
  t_map.header.frame_id = "map";
  t_map.child_frame_id = "odom";
  t_map.transform.translation.x = 0.0;
  t_map.transform.translation.y = 0.0;
  t_map.transform.translation.z = 0.0;
  t_map.transform.rotation.x = 0.0;
  t_map.transform.rotation.y = 0.0;
  t_map.transform.rotation.z = 0.0;
  t_map.transform.rotation.w = 1.0;
  static_tf_broadcaster_->sendTransform(t_map);

  RCLCPP_INFO(this->get_logger(),
    "Crazyflie mapper initialized: %.1fx%.1fm (%dx%d cells) @ %.2fm/cell, origin (%.1f, %.1f)",
    map_size_x_, map_size_y_, map_width_, map_height_, map_resolution_,
    map_origin_x_, map_origin_y_);
  RCLCPP_INFO(this->get_logger(),
    "Avoidance dist %.2fm, max weight %d, max obstacle range %.2fm",
    avoidance_distance_, max_avoidance_weight_, max_obstacle_range_);

  publish_map();
}

void MapperNode::pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  position_[0] = msg->pose.position.x;
  position_[1] = msg->pose.position.y;
  position_[2] = msg->pose.position.z;

  const auto & q = msg->pose.orientation;
  double roll, pitch, yaw;
  euler_from_quaternion(q.x, q.y, q.z, q.w, roll, pitch, yaw);
  angles_[0] = roll;
  angles_[1] = pitch;
  angles_[2] = yaw;

  position_update_ = true;
}

void MapperNode::scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
  if (msg->ranges.size() < 4) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "LaserScan has %zu ranges; expected at least 4 (back,right,front,left)",
      msg->ranges.size());
    return;
  }
  ranges_ = std::vector<float>(msg->ranges.begin(), msg->ranges.begin() + 4);
  range_max_ = msg->range_max;

  if (!position_update_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "No pose update yet, skipping scan");
    return;
  }

  const auto obstacles = rotate_and_create_points();

  const int position_x_map =
    static_cast<int>((position_[0] - map_origin_x_) / map_resolution_);
  const int position_y_map =
    static_cast<int>((position_[1] - map_origin_y_) / map_resolution_);

  if (position_x_map < 0 || position_x_map >= map_width_ ||
      position_y_map < 0 || position_y_map >= map_height_)
  {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "Robot outside map bounds at (%d, %d); skipping scan",
      position_x_map, position_y_map);
    return;
  }

  for (const auto & p : obstacles) {
    const int point_x = static_cast<int>((p[0] - map_origin_x_) / map_resolution_);
    const int point_y = static_cast<int>((p[1] - map_origin_y_) / map_resolution_);

    if (point_x < 0 || point_x >= map_width_ || point_y < 0 || point_y >= map_height_) {
      continue;
    }

    const auto ray = bresenham_line(position_x_map, position_y_map, point_x, point_y);

    // Free cells along the ray — only promote unknown/free, never overwrite weighted/occupied
    for (const auto & [lx, ly] : ray) {
      if (lx < 0 || lx >= map_width_ || ly < 0 || ly >= map_height_) {
        continue;
      }
      const size_t idx = static_cast<size_t>(ly) * static_cast<size_t>(map_width_) +
                         static_cast<size_t>(lx);
      if (map_[idx] == -1 || map_[idx] == 0) {
        map_[idx] = 0;
      }
    }

    // Mark obstacle cell
    const size_t obs_idx = static_cast<size_t>(point_y) *
                           static_cast<size_t>(map_width_) +
                           static_cast<size_t>(point_x);
    map_[obs_idx] = 100;

    const int obstacle_distance_cells =
      std::max<int>(0, static_cast<int>(ray.size()) - 1);
    apply_ray_weights(ray, obstacle_distance_cells);
  }
}

void MapperNode::publish_map()
{
  nav_msgs::msg::OccupancyGrid msg;
  msg.header.stamp = this->get_clock()->now();
  msg.header.frame_id = "map";
  msg.info.resolution = static_cast<float>(map_resolution_);
  msg.info.width = static_cast<uint32_t>(map_width_);
  msg.info.height = static_cast<uint32_t>(map_height_);
  msg.info.origin.position.x = map_origin_x_;
  msg.info.origin.position.y = map_origin_y_;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.w = 1.0;
  msg.data = map_;
  map_pub_->publish(msg);
}

std::vector<std::array<double, 3>> MapperNode::rotate_and_create_points() const
{
  std::vector<std::array<double, 3>> data;
  const auto & o = position_;
  const double roll  = angles_[0];
  const double pitch = angles_[1];
  const double yaw   = angles_[2];

  const float r_back  = ranges_[0];
  const float r_right = ranges_[1];
  const float r_front = ranges_[2];
  const float r_left  = ranges_[3];

  auto valid = [this](float r) {
    return std::isfinite(r) && r != 0.0f &&
           r < static_cast<float>(range_max_) &&
           r < static_cast<float>(max_obstacle_range_);
  };

  if (valid(r_left)) {
    std::array<double, 3> left{{o[0], o[1] + r_left, o[2]}};
    data.push_back(rotate(roll, pitch, yaw, o, left));
  }
  if (valid(r_right)) {
    std::array<double, 3> right{{o[0], o[1] - r_right, o[2]}};
    data.push_back(rotate(roll, pitch, yaw, o, right));
  }
  if (valid(r_front)) {
    std::array<double, 3> front{{o[0] + r_front, o[1], o[2]}};
    data.push_back(rotate(roll, pitch, yaw, o, front));
  }
  if (valid(r_back)) {
    std::array<double, 3> back{{o[0] - r_back, o[1], o[2]}};
    data.push_back(rotate(roll, pitch, yaw, o, back));
  }
  return data;
}

std::array<double, 3> MapperNode::rotate(
  double roll, double pitch, double yaw,
  const std::array<double, 3> & origin,
  const std::array<double, 3> & point)
{
  const double cr = std::cos(roll),  sr = std::sin(roll);
  const double cp = std::cos(pitch), sp = std::sin(pitch);
  const double cy = std::cos(yaw),   sy = std::sin(yaw);

  // Combined R = Rr * Rp * Ry (matches the Python ordering)
  const double m00 = cp * cy;
  const double m01 = -cp * sy;
  const double m02 = sp;
  const double m10 = sr * sp * cy + cr * sy;
  const double m11 = -sr * sp * sy + cr * cy;
  const double m12 = -sr * cp;
  const double m20 = -cr * sp * cy + sr * sy;
  const double m21 = cr * sp * sy + sr * cy;
  const double m22 = cr * cp;

  const double dx = point[0] - origin[0];
  const double dy = point[1] - origin[1];
  const double dz = point[2] - origin[2];

  return {{
    m00 * dx + m01 * dy + m02 * dz + origin[0],
    m10 * dx + m11 * dy + m12 * dz + origin[1],
    m20 * dx + m21 * dy + m22 * dz + origin[2]
  }};
}

double MapperNode::yaw_from_quaternion(double x, double y, double z, double w)
{
  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

void MapperNode::euler_from_quaternion(
  double x, double y, double z, double w,
  double & roll, double & pitch, double & yaw)
{
  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = 2.0 * (w * y - z * x);
  if (std::abs(sinp) >= 1.0) {
    pitch = std::copysign(M_PI / 2.0, sinp);
  } else {
    pitch = std::asin(sinp);
  }

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  yaw = std::atan2(siny_cosp, cosy_cosp);
}

std::vector<MapperNode::Cell> MapperNode::bresenham_line(int x0, int y0, int x1, int y1)
{
  std::vector<Cell> cells;
  const int dx = std::abs(x1 - x0);
  const int dy = -std::abs(y1 - y0);
  const int sx = (x0 < x1) ? 1 : -1;
  const int sy = (y0 < y1) ? 1 : -1;
  int err = dx + dy;
  int x = x0, y = y0;
  while (true) {
    cells.emplace_back(x, y);
    if (x == x1 && y == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
  return cells;
}

void MapperNode::apply_ray_weights(const std::vector<Cell> & ray_cells, int obstacle_distance_cells)
{
  if (ray_cells.empty()) {
    return;
  }
  const int K = std::min(6, std::max(1, obstacle_distance_cells));
  const int ray_len = static_cast<int>(ray_cells.size());
  const int start = std::max(0, ray_len - K - 1);
  const int end = std::max(0, ray_len - 1);  // exclude the obstacle cell itself
  if (start >= end) {
    return;
  }

  const int window_len = end - start;
  for (int i = 0; i < window_len; ++i) {
    const auto & [cx, cy] = ray_cells[start + i];
    if (cx < 0 || cx >= map_width_ || cy < 0 || cy >= map_height_) {
      continue;
    }
    const size_t idx = static_cast<size_t>(cy) * static_cast<size_t>(map_width_) +
                       static_cast<size_t>(cx);
    const int8_t current = map_[idx];
    if (current != -1 && current != 0) {
      continue;  // never overwrite occupied or previously weighted cells
    }
    int w = static_cast<int>(std::round(
      static_cast<double>(i + 1) * max_avoidance_weight_ / static_cast<double>(K)));
    w = std::max(1, std::min(w, max_avoidance_weight_));
    map_[idx] = static_cast<int8_t>(w);
  }
}

}  // namespace crazyflie_mapper

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<crazyflie_mapper::MapperNode>());
  rclcpp::shutdown();
  return 0;
}
