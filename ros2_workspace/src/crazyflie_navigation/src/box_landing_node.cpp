#include "crazyflie_navigation/box_landing_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace crazyflie_navigation
{

BoxLandingNode::BoxLandingNode()
: rclcpp::Node("box_landing_node"),
  pose_(nullptr),
  phase_(LandingPhase::IDLE)
{
  // Parameters
  enabled_ = this->declare_parameter<bool>("enabled", false);
  flight_height_ = this->declare_parameter<double>("flight_height", 0.5);
  search_span_ = this->declare_parameter<double>("search_span_m", 0.6);
  search_step_ = this->declare_parameter<double>("search_step_m", 0.15);
  vz_max_ = this->declare_parameter<double>("max_descent_rate", 0.08);
  min_hover_ = this->declare_parameter<double>("min_hover_height", 0.12);
  min_edge_points_ = this->declare_parameter<int>("min_edge_points", 20);
  goal_tolerance_ = this->declare_parameter<double>("goal_tolerance", 0.15);
  box_landing_speed_ = this->declare_parameter<double>("box_landing_speed", 1.0);
  search_scale_ = this->declare_parameter<double>("search_scale", 1.0);
  edge_delta_threshold_ = this->declare_parameter<double>("edge_delta_threshold", 0.08);

  // IO — pose comes from EKF
  pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
    "/ekf_pose", 10,
    std::bind(&BoxLandingNode::pose_cb, this, std::placeholders::_1));

  const std::array<std::string, 5> range_topics = {
    "/crazyflie/range/front",
    "/crazyflie/range/back",
    "/crazyflie/range/left",
    "/crazyflie/range/right",
    "/crazyflie/range/down",
  };
  for (size_t i = 0; i < 5; ++i) {
    RangeDir dir = static_cast<RangeDir>(i);
    range_subs_[i] = this->create_subscription<sensor_msgs::msg::Range>(
      range_topics[i], 10,
      [this, dir](const sensor_msgs::msg::Range::SharedPtr msg) {
        this->range_cb(msg, dir);
      });
  }

  cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
  estop_pub_ = this->create_publisher<std_msgs::msg::Bool>("/crazyflie/emergency_stop", 10);
  status_pub_ = this->create_publisher<std_msgs::msg::String>("/box_landing/status", 10);
  goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/goal_pose", 10);

  autonomous_nav_client_ = this->create_client<std_srvs::srv::SetBool>("/enable_autonomous");

  enable_srv_ = this->create_service<std_srvs::srv::SetBool>(
    "/enable_box_landing",
    std::bind(&BoxLandingNode::enable_cb, this,
              std::placeholders::_1, std::placeholders::_2));

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&BoxLandingNode::loop, this));

  RCLCPP_INFO(this->get_logger(), "box_landing_node ready (disabled by default)");
}

void BoxLandingNode::pose_cb(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  bool was_none = (pose_ == nullptr);
  pose_ = msg;
  if (was_none) {
    RCLCPP_INFO(this->get_logger(),
      "Received first pose: (%.2f, %.2f, %.2f)",
      msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  }
}

void BoxLandingNode::range_cb(
  const sensor_msgs::msg::Range::SharedPtr msg, RangeDir dir)
{
  size_t idx = static_cast<size_t>(dir);
  previous_range_[idx] = latest_range_[idx];
  latest_range_[idx] = msg->range;

  // Run edge detection on each new sample while grid searching
  detect_edges();
}

void BoxLandingNode::detect_edges()
{
  if (!enabled_ || phase_ != LandingPhase::GRID_SEARCH || pose_ == nullptr) {
    return;
  }

  // Edge on the downward sensor = step up/down onto/off a box.
  // Significant delta on any sensor between consecutive samples flags an edge crossing.
  bool edge_triggered = false;
  for (size_t i = 0; i < 5; ++i) {
    if (!latest_range_[i].has_value() || !previous_range_[i].has_value()) {
      continue;
    }
    double delta = std::abs(latest_range_[i].value() - previous_range_[i].value());
    if (delta > edge_delta_threshold_) {
      edge_triggered = true;
      break;
    }
  }

  if (!edge_triggered) {
    return;
  }

  double x = pose_->pose.position.x;
  double y = pose_->pose.position.y;
  edge_points_.emplace_back(x, y);
  RCLCPP_INFO(this->get_logger(),
    "EDGE POINT #%zu recorded at (%.3f, %.3f)", edge_points_.size(), x, y);

  if (static_cast<int>(edge_points_.size()) >= min_edge_points_) {
    RCLCPP_INFO(this->get_logger(),
      "Collected %zu edge points (>= %d) - stopping grid search",
      edge_points_.size(), min_edge_points_);
    compute_and_navigate_to_centroid();
  }
}

void BoxLandingNode::enable_cb(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
  std::shared_ptr<std_srvs::srv::SetBool::Response> response)
{
  enabled_ = request->data;
  response->success = true;
  response->message = std::string("box_landing_node ") +
                      (enabled_ ? "enabled" : "disabled");
  RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());

  if (enabled_) {
    edge_points_.clear();
    box_centroid_.reset();
    grid_center_.reset();
    grid_sequence_.clear();
    grid_start_time_.reset();

    if (pose_ != nullptr) {
      double x = pose_->pose.position.x;
      double y = pose_->pose.position.y;
      edge_points_.emplace_back(x, y);
      RCLCPP_INFO(this->get_logger(),
        "Box landing enabled - using current position as EDGE POINT #1: (%.3f, %.3f)",
        x, y);
      grid_center_ = std::make_pair(x, y);
      grid_sequence_ = generate_grid_sequence();
      grid_start_time_ = this->get_clock()->now().seconds();
      phase_ = LandingPhase::GRID_SEARCH;
      publish_status("grid_searching");
      double total_duration = 0.0;
      for (const auto & seg : grid_sequence_) {
        total_duration += seg.duration;
      }
      RCLCPP_INFO(this->get_logger(),
        "Starting grid search immediately - center at (%.3f, %.3f), %zu movements, total duration=%.1fs",
        grid_center_->first, grid_center_->second, grid_sequence_.size(), total_duration);
    } else {
      RCLCPP_WARN(this->get_logger(),
        "Box landing enabled but no pose available yet - waiting");
      phase_ = LandingPhase::IDLE;
      publish_status("waiting_for_pose");
    }
  } else {
    phase_ = LandingPhase::IDLE;
    cmd_pub_->publish(geometry_msgs::msg::Twist());
    RCLCPP_INFO(this->get_logger(), "Box landing disabled - stopping");
  }
}

void BoxLandingNode::publish_status(const std::string & status)
{
  std_msgs::msg::String msg;
  msg.data = status;
  status_pub_->publish(msg);
}

std::vector<GridSegment> BoxLandingNode::generate_grid_sequence()
{
  std::vector<GridSegment> sequence;
  if (!grid_center_.has_value()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot generate grid: grid_center is None");
    return sequence;
  }

  double half_span = search_span_ / 2.0;
  double step = search_step_;

  int num_cols = static_cast<int>(search_span_ / step) + 1;
  int num_rows = static_cast<int>(search_span_ / step) + 1;

  double horizontal_duration = step / box_landing_speed_ * search_scale_;
  double vertical_duration = step / box_landing_speed_ * search_scale_;
  double full_span_duration = search_span_ / box_landing_speed_ * search_scale_;
  double half_span_duration = half_span / box_landing_speed_ * search_scale_;

  // Move up (positive Y)
  sequence.push_back({0.0, 1.0, half_span_duration});
  // Move left (negative X)
  sequence.push_back({-1.0, 0.0, half_span_duration});

  // Horizontal raster
  for (int row = 0; row < num_rows; ++row) {
    if (row % 2 == 0) {
      sequence.push_back({1.0, 0.0, full_span_duration});
    } else {
      sequence.push_back({-1.0, 0.0, full_span_duration});
    }
    if (row < num_rows - 1) {
      sequence.push_back({0.0, -1.0, vertical_duration});
    }
  }

  if ((num_rows - 1) % 2 == 0) {
    sequence.push_back({-1.0, 0.0, full_span_duration});
  }

  // Perpendicular raster
  for (int col = 0; col < num_cols; ++col) {
    if (col % 2 == 0) {
      sequence.push_back({0.0, 1.0, full_span_duration});
    } else {
      sequence.push_back({0.0, -1.0, full_span_duration});
    }
    if (col < num_cols - 1) {
      sequence.push_back({1.0, 0.0, horizontal_duration});
    }
  }

  return sequence;
}

void BoxLandingNode::call_autonomous_nav(bool enable)
{
  if (!autonomous_nav_client_->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_WARN(this->get_logger(), "Autonomous navigation service not available");
    return;
  }
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = enable;
  autonomous_nav_client_->async_send_request(request);
  RCLCPP_INFO(this->get_logger(),
    "Called autonomous navigation: %s", enable ? "ENABLE" : "DISABLE");
}

void BoxLandingNode::compute_and_navigate_to_centroid()
{
  if (edge_points_.size() < 3) {
    RCLCPP_WARN(this->get_logger(),
      "Not enough edge points: %zu < %d", edge_points_.size(), min_edge_points_);
    return;
  }

  double sum_x = 0.0, sum_y = 0.0;
  for (const auto & p : edge_points_) {
    sum_x += p.first;
    sum_y += p.second;
  }
  double cx = sum_x / static_cast<double>(edge_points_.size());
  double cy = sum_y / static_cast<double>(edge_points_.size());
  box_centroid_ = std::make_pair(cx, cy);

  RCLCPP_INFO(this->get_logger(),
    "Box centroid computed: (%.3f, %.3f) from %zu edge points",
    cx, cy, edge_points_.size());

  geometry_msgs::msg::PoseStamped goal_msg;
  goal_msg.header.stamp = this->get_clock()->now();
  goal_msg.header.frame_id = "world";
  goal_msg.pose.position.x = cx;
  goal_msg.pose.position.y = cy;
  goal_msg.pose.position.z = flight_height_;
  goal_msg.pose.orientation.w = 1.0;
  goal_pub_->publish(goal_msg);
  RCLCPP_INFO(this->get_logger(), "Published goal_pose to (%.3f, %.3f)", cx, cy);

  call_autonomous_nav(true);

  phase_ = LandingPhase::NAVIGATE;
  publish_status("navigating");
  RCLCPP_INFO(this->get_logger(),
    "Transitioning to NAVIGATE phase - autonomous navigation enabled");
}

void BoxLandingNode::loop()
{
  if (!enabled_ || pose_ == nullptr) {
    return;
  }

  if (phase_ == LandingPhase::GRID_SEARCH) {
    if (!grid_start_time_.has_value()) {
      RCLCPP_WARN(this->get_logger(), "GRID_SEARCH: No start time set");
      return;
    }
    if (grid_sequence_.empty()) {
      RCLCPP_ERROR(this->get_logger(), "GRID_SEARCH: No sequence available!");
      phase_ = LandingPhase::WAIT_COMPLETE;
      publish_status("completed");
      wait_start_time_ = this->get_clock()->now().seconds();
      cmd_pub_->publish(geometry_msgs::msg::Twist());
      return;
    }

    double now = this->get_clock()->now().seconds();
    double elapsed = now - grid_start_time_.value();
    double total_duration = 0.0;
    for (const auto & seg : grid_sequence_) {
      total_duration += seg.duration;
    }

    if (elapsed >= total_duration) {
      RCLCPP_INFO(this->get_logger(),
        "GRID_SEARCH: Sequence complete - %zu edges detected", edge_points_.size());

      if (edge_points_.size() >= 3) {
        if (static_cast<int>(edge_points_.size()) >= min_edge_points_) {
          RCLCPP_INFO(this->get_logger(),
            "Sufficient edges collected (%zu >= %d)",
            edge_points_.size(), min_edge_points_);
        } else {
          RCLCPP_WARN(this->get_logger(),
            "Only %zu edges collected (target: %d), but attempting navigation with available points",
            edge_points_.size(), min_edge_points_);
        }
        compute_and_navigate_to_centroid();
      } else {
        RCLCPP_WARN(this->get_logger(),
          "Insufficient edges for navigation: %zu < 3 (minimum required)",
          edge_points_.size());
        phase_ = LandingPhase::WAIT_COMPLETE;
        publish_status("completed");
        wait_start_time_ = now;
        cmd_pub_->publish(geometry_msgs::msg::Twist());
      }
      return;
    }

    // Find current segment in sequence
    double cumulative_time = 0.0;
    for (const auto & seg : grid_sequence_) {
      if (elapsed < cumulative_time + seg.duration) {
        geometry_msgs::msg::Twist v;
        v.linear.x = seg.vx_unit * box_landing_speed_;
        v.linear.y = seg.vy_unit * box_landing_speed_;
        double zerr = flight_height_ - pose_->pose.position.z;
        v.linear.z = std::max(std::min(1.0 * zerr, 0.2), -0.2);
        v.angular.z = 0.0;
        cmd_pub_->publish(v);
        return;
      }
      cumulative_time += seg.duration;
    }

    cmd_pub_->publish(geometry_msgs::msg::Twist());
    return;
  }

  if (phase_ == LandingPhase::NAVIGATE) {
    if (!box_centroid_.has_value()) {
      RCLCPP_ERROR(this->get_logger(), "NAVIGATE: No centroid set!");
      return;
    }
    double cx = box_centroid_->first;
    double cy = box_centroid_->second;
    double px = pose_->pose.position.x;
    double py = pose_->pose.position.y;
    double dist = std::hypot(cx - px, cy - py);

    if (dist < goal_tolerance_) {
      RCLCPP_INFO(this->get_logger(),
        "NAVIGATE: Reached box centroid! Distance=%.3fm < %.3fm",
        dist, goal_tolerance_);
      call_autonomous_nav(false);
      phase_ = LandingPhase::WAIT_COMPLETE;
      publish_status("completing");
      wait_start_time_ = this->get_clock()->now().seconds();
      cmd_pub_->publish(geometry_msgs::msg::Twist());
      RCLCPP_INFO(this->get_logger(),
        "Transitioning to WAIT_COMPLETE - hovering for 1 second");
    }
    return;
  }

  if (phase_ == LandingPhase::WAIT_COMPLETE) {
    if (!wait_start_time_.has_value()) {
      RCLCPP_WARN(this->get_logger(), "WAIT_COMPLETE: No wait start time set");
      return;
    }
    double elapsed = this->get_clock()->now().seconds() - wait_start_time_.value();
    cmd_pub_->publish(geometry_msgs::msg::Twist());
    if (elapsed >= 1.0) {
      RCLCPP_INFO(this->get_logger(),
        "WAIT_COMPLETE: 1 second elapsed - delegating back to mode_manager");
      publish_status("completed");
      phase_ = LandingPhase::STOP;
      enabled_ = false;
      RCLCPP_INFO(this->get_logger(), "Box landing sequence complete!");
    }
    return;
  }
}

}  // namespace crazyflie_navigation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<crazyflie_navigation::BoxLandingNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
