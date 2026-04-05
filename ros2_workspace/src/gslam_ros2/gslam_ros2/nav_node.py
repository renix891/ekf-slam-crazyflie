#!/usr/bin/env python3
"""
Autonomous Navigation Node

Orchestrates scan-plan-move loop for autonomous navigation:
1. Rotate 90 degrees to scan surroundings
2. Request path planning
3. Select next waypoint from path
4. Move to waypoint
5. Repeat until goal reached

Subscribes to:
  /odom (nav_msgs/Odometry) - current position
  /position_target_reached (std_msgs/Bool) - waypoint arrival confirmation
  /planned_path (nav_msgs/Path) - path from planner
  /goal_pose (geometry_msgs/PoseStamped) - navigation goal

Publishes:
  /cmd_pos (geometry_msgs/PointStamped) - position commands

Service Clients:
  /trigger_replanning (std_srvs/Trigger) - request path planning
  /scan_surround (std_srvs/Trigger) - rotate 90 degrees
"""

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

import math
import threading
import time
from typing import Optional, Tuple


class NavNode(Node):
    def __init__(self):
        super().__init__('nav_node')

        # Parameters
        self.declare_parameter('max_waypoint_distance', 1.0)
        self.declare_parameter('enabled', False)
        self.declare_parameter('flight_height', 0.5)

        self.max_waypoint_distance = self.get_parameter('max_waypoint_distance').value
        self.enabled = self.get_parameter('enabled').value
        self.flight_height = self.get_parameter('flight_height').value

        # State
        self.current_odom = None
        self.goal_pose = None
        self.current_path = None
        self.position_reached = False
        self.goal_reached = False

        # Locks
        self.path_lock = threading.Lock()
        self.position_reached_lock = threading.Lock()

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.position_reached_sub = self.create_subscription(
            Bool, '/position_target_reached', self.position_reached_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publisher
        self.cmd_pos_pub = self.create_publisher(PointStamped, '/cmd_pos', 10)

        # Service clients
        self.replan_client = self.create_client(Trigger, '/trigger_replanning')
        self.scan_client = self.create_client(Trigger, '/scan_surround')

        # Navigation thread
        self.nav_thread = None
        self.nav_running = False

        self.get_logger().info(f'Navigation node initialized')
        self.get_logger().info(f'Max waypoint distance: {self.max_waypoint_distance}m')
        self.get_logger().info(f'Enabled: {self.enabled}')

        # Start navigation loop if enabled
        if self.enabled:
            self.start_navigation()

    def odom_callback(self, msg):
        """Update current odometry."""
        self.current_odom = msg

    def position_reached_callback(self, msg):
        """Handle position target reached confirmation."""
        with self.position_reached_lock:
            if msg.data:
                self.position_reached = True
                self.get_logger().debug('Position target reached')

    def path_callback(self, msg):
        """Update current path."""
        with self.path_lock:
            self.current_path = msg
            self.get_logger().debug(f'Received path with {len(msg.poses)} waypoints')

    def goal_callback(self, msg):
        """Handle new goal."""
        self.goal_pose = msg
        self.goal_reached = False
        self.get_logger().info(f'New goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        # Start navigation if not already running
        if self.enabled and not self.nav_running:
            self.start_navigation()

    def start_navigation(self):
        """Start the navigation loop in a separate thread."""
        if self.nav_running:
            self.get_logger().warning('Navigation already running')
            return

        self.nav_running = True
        self.nav_thread = threading.Thread(target=self.navigation_loop, daemon=True)
        self.nav_thread.start()
        self.get_logger().info('Navigation loop started')

    def stop_navigation(self):
        """Stop the navigation loop."""
        self.nav_running = False
        if self.nav_thread:
            self.nav_thread.join(timeout=5.0)
        self.get_logger().info('Navigation loop stopped')

    def navigation_loop(self):
        """Main navigation loop: scan -> plan -> select waypoint -> move -> repeat."""
        while rclpy.ok() and self.nav_running and not self.goal_reached:
            # Check if we have necessary data
            if self.current_odom is None:
                self.get_logger().warning('Waiting for odometry...')
                time.sleep(1.0)
                continue

            if self.goal_pose is None:
                self.get_logger().warning('Waiting for goal...')
                time.sleep(1.0)
                continue

            # 1. Scan surroundings (rotate 90 degrees)
            self.get_logger().info('Step 1: Scanning surroundings...')
            if not self.call_scan_surround():
                self.get_logger().error('Scan failed, retrying...')
                time.sleep(1.0)
                continue

            # 2. Request path planning
            self.get_logger().info('Step 2: Requesting path planning...')
            if not self.call_replan():
                self.get_logger().error('Planning failed, retrying...')
                time.sleep(1.0)
                continue

            # 3. Wait for updated path
            time.sleep(0.5)  # Give planner time to publish path

            # 4. Select waypoint from path
            waypoint = self.select_waypoint_from_path()
            if waypoint is None:
                self.get_logger().warning('No valid waypoint found, retrying...')
                time.sleep(1.0)
                continue

            self.get_logger().info(f'Step 3: Selected waypoint ({waypoint[0]:.2f}, {waypoint[1]:.2f})')

            # 5. Move to waypoint
            self.publish_waypoint(waypoint)

            # 6. Wait for arrival
            self.get_logger().info('Step 4: Moving to waypoint...')
            if not self.wait_for_position_reached(timeout=30.0):
                self.get_logger().warning('Waypoint timeout, continuing anyway...')

            # 7. Check if goal reached (waypoint is goal)
            if self.is_waypoint_goal(waypoint):
                self.goal_reached = True
                self.get_logger().info('Goal reached!')
                break

        self.nav_running = False
        self.get_logger().info('Navigation loop exited')

    def call_scan_surround(self) -> bool:
        """Call scan_surround service to rotate 90 degrees."""
        if not self.scan_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Scan service not available')
            return False

        request = Trigger.Request()
        future = self.scan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None and future.result().success:
            self.get_logger().debug('Scan service succeeded')
            return True
        else:
            self.get_logger().error('Scan service failed')
            return False

    def call_replan(self) -> bool:
        """Call trigger_replanning service."""
        if not self.replan_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Replan service not available')
            return False

        request = Trigger.Request()
        future = self.replan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None and future.result().success:
            self.get_logger().debug('Replan service succeeded')
            return True
        else:
            self.get_logger().error('Replan service failed')
            return False

    def select_waypoint_from_path(self) -> Optional[Tuple[float, float]]:
        """
        Select next waypoint from path.
        Returns furthest point within max_waypoint_distance.
        """
        with self.path_lock:
            if self.current_path is None or len(self.current_path.poses) == 0:
                return None

            path = self.current_path

        if self.current_odom is None:
            return None

        current_x = self.current_odom.pose.pose.position.x
        current_y = self.current_odom.pose.pose.position.y

        # Find furthest reachable waypoint
        selected_waypoint = None
        max_distance_found = 0.0

        for pose in path.poses:
            wx = pose.pose.position.x
            wy = pose.pose.position.y

            distance = math.hypot(wx - current_x, wy - current_y)

            # Check if within max distance
            if distance <= self.max_waypoint_distance and distance > max_distance_found:
                selected_waypoint = (wx, wy)
                max_distance_found = distance

        # If no waypoint within max distance, use the closest one from the path
        if selected_waypoint is None and len(path.poses) > 0:
            # Just use the first waypoint
            pose = path.poses[0]
            selected_waypoint = (pose.pose.position.x, pose.pose.position.y)
            self.get_logger().warning(f'No waypoint within max distance, using first: {selected_waypoint}')

        return selected_waypoint

    def publish_waypoint(self, waypoint: Tuple[float, float]):
        """Publish waypoint as position command."""
        # Reset position reached flag
        with self.position_reached_lock:
            self.position_reached = False

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.point.x = waypoint[0]
        msg.point.y = waypoint[1]
        msg.point.z = self.flight_height

        self.cmd_pos_pub.publish(msg)
        self.get_logger().debug(f'Published waypoint: ({waypoint[0]:.2f}, {waypoint[1]:.2f})')

    def wait_for_position_reached(self, timeout: float = 30.0) -> bool:
        """Wait for position_target_reached signal."""
        start_time = time.time()

        while rclpy.ok():
            with self.position_reached_lock:
                if self.position_reached:
                    return True

            # Check timeout
            if time.time() - start_time > timeout:
                return False

            time.sleep(0.1)

        return False

    def is_waypoint_goal(self, waypoint: Tuple[float, float]) -> bool:
        """Check if waypoint is the goal."""
        if self.goal_pose is None:
            return False

        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        distance = math.hypot(waypoint[0] - goal_x, waypoint[1] - goal_y)

        # Consider waypoint as goal if very close (within 0.1m)
        return distance < 0.1


def main(args=None):
    rclpy.init(args=args)
    node = NavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_navigation()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
