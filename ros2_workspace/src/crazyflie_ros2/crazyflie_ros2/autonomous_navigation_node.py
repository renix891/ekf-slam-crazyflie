"""
Autonomous Navigation Node for Crazyflie (Waypoint Following with Scan-at-Waypoint)

Subscribes to path waypoints from a separate path planning node and follows them.
At each waypoint, the drone stops and rotates 90 degrees to scan the environment.
Enabled/disabled via service by mode_manager_node.

Subscribes to:
  /planned_path (nav_msgs/Path) - waypoints from path planner
  /crazyflie/pose (geometry_msgs/PoseStamped)

Publishes:
  /cmd_vel (geometry_msgs/Twist)
"""

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import SetBool

import numpy as np
import math


class AutonomousNavigationNode(Node):
    """ROS2 node for waypoint following with scan-at-waypoint behavior."""

    def __init__(self):
        super().__init__('autonomous_navigation_node')

        # Parameters
        self.declare_parameter('waypoint_tolerance', 0.2)
        self.declare_parameter('scanning_yaw_rate', 1.0)
        self.declare_parameter('flight_height', 0.3)
        self.declare_parameter('rotation_tolerance', 0.1)  # rad
        self.declare_parameter('yaw_kp', 2.0)  # Proportional gain for yaw control
        self.declare_parameter('vz_kp', 1.0)  # Altitude control gain

        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.scanning_yaw_rate = self.get_parameter('scanning_yaw_rate').value
        self.flight_height = self.get_parameter('flight_height').value
        self.rotation_tolerance = float(self.get_parameter('rotation_tolerance').value)
        self.yaw_kp = float(self.get_parameter('yaw_kp').value)
        self.vz_kp = float(self.get_parameter('vz_kp').value)

        # State
        self.enabled = False
        self.current_pose = None
        self.planned_path = []
        self.current_waypoint_idx = 0
        self.waypoint_state = 'MOVING'  # 'MOVING' or 'SCANNING'
        self.target_yaw = None  # Target yaw for rotation

        # Subscribers
        self.pose_sub = self.create_subscription(PoseStamped, '/crazyflie/pose', self.pose_callback, 10)
        self.path_sub = self.create_subscription(Path, '/planned_path', self.path_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Services
        self.enable_srv = self.create_service(SetBool, '/enable_autonomous', self.enable_callback)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.status_timer = self.create_timer(1.0, self.status_loop)  # 1 Hz status updates

        self.get_logger().info('Autonomous Navigation Node initialized (disabled)')

    # ----- Callbacks -----
    def pose_callback(self, msg):
        was_none = self.current_pose is None
        self.current_pose = msg
        if was_none:
            self.get_logger().debug(
                f'Received first pose: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def path_callback(self, msg: Path):
        """Receive planned path from path planning node."""
        if len(msg.poses) > 0:
            self.planned_path = msg.poses
            self.current_waypoint_idx = 0
            self.get_logger().debug(f'Received path with {len(msg.poses)} waypoints')

    def enable_callback(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f"Autonomous navigation {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)
        if not self.enabled:
            self._publish_zero_velocity()
            self.current_waypoint_idx = 0
            self.waypoint_state = 'MOVING'
            self.target_yaw = None
        return response

    # ----- Utility Functions -----
    def _quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle (rotation around Z-axis)."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _world_to_body_frame(self, v_world_xy, current_yaw):
        """
        Convert world-frame velocity to body-frame velocity.

        Args:
            v_world_xy: np.array([vx_world, vy_world]) - velocity in world frame
            current_yaw: current yaw angle in radians

        Returns:
            np.array([vx_body, vy_body]) - velocity in body frame
        """
        # Rotation matrix from world to body frame
        R = np.array([[ math.cos(-current_yaw), -math.sin(-current_yaw) ],
                      [ math.sin(-current_yaw),  math.cos(-current_yaw) ]])
        return R @ v_world_xy

    def rotate_to_target(self, target_yaw: float) -> bool:
        """
        Rotate to target yaw angle using proportional control.
        Yaw rate is proportional to error, clamped to max scanning_yaw_rate.

        Returns True if target reached, False otherwise.
        """
        if self.current_pose is None:
            return False

        current_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
        yaw_error = self._normalize_angle(target_yaw - current_yaw)

        # Check if rotation complete
        if abs(yaw_error) < self.rotation_tolerance:
            self.get_logger().info(
                f'Rotation complete - error={math.degrees(yaw_error):.1f}°')
            self._publish_zero_velocity()
            return True

        # Proportional control: yaw_rate_cmd = kp * error, clamped to max
        yaw_rate_cmd = self.yaw_kp * yaw_error
        yaw_rate_cmd = max(min(yaw_rate_cmd, self.scanning_yaw_rate), -self.scanning_yaw_rate)

        # Send rotation command
        cmd = Twist()
        cmd.angular.z = yaw_rate_cmd

        # Altitude hold
        z_err = self.flight_height - self.current_pose.pose.position.z
        cmd.linear.z = self.vz_kp * z_err

        self.cmd_vel_pub.publish(cmd)
        return False

    # ----- Control Loop -----
    def control_loop(self):
        """Main control loop - follows waypoints with scanning rotations."""
        if not self.enabled or self.current_pose is None:
            return

        # If no path, hover in place
        if not self.planned_path:
            self._publish_zero_velocity()
            return

        # Check if we've reached the end of the path
        if self.current_waypoint_idx >= len(self.planned_path):
            self._publish_zero_velocity()
            return

        # State machine: MOVING or SCANNING
        if self.waypoint_state == 'MOVING':
            # Get current waypoint
            waypoint = self.planned_path[self.current_waypoint_idx]
            dx = waypoint.pose.position.x - self.current_pose.pose.position.x
            dy = waypoint.pose.position.y - self.current_pose.pose.position.y
            dist = math.hypot(dx, dy)

            # Check if we've reached the current waypoint
            # Use tighter tolerance for final waypoint (goal_pose)
            is_final_waypoint = (self.current_waypoint_idx == len(self.planned_path) - 1)
            tolerance = self.waypoint_tolerance / 5.0 if is_final_waypoint else self.waypoint_tolerance

            if dist < tolerance:
                # Waypoint reached - transition to SCANNING state
                current_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
                self.target_yaw = self._normalize_angle(current_yaw + math.pi / 2.0)  # Rotate 90 degrees
                self.waypoint_state = 'SCANNING'
                self.get_logger().info(
                    f'Waypoint {self.current_waypoint_idx} reached - starting scan rotation to '
                    f'{math.degrees(self.target_yaw):.1f}°')
                return

            # Move towards current waypoint
            # Calculate world-frame direction
            v_world_xy = np.array([dx, dy], dtype=float) / max(dist, 1e-6)

            # Convert to body-frame based on current heading
            current_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
            v_body_xy = self._world_to_body_frame(v_world_xy, current_yaw)

            # Create velocity command
            cmd = Twist()
            cmd.linear.x = float(v_body_xy[0])
            cmd.linear.y = float(v_body_xy[1])

            # Altitude hold
            z_err = self.flight_height - self.current_pose.pose.position.z
            cmd.linear.z = self.vz_kp * z_err

            self.cmd_vel_pub.publish(cmd)

        elif self.waypoint_state == 'SCANNING':
            # Rotate 90 degrees at waypoint
            if self.rotate_to_target(self.target_yaw):
                # Rotation complete - advance to next waypoint
                self.current_waypoint_idx += 1
                self.waypoint_state = 'MOVING'
                self.get_logger().info(
                    f'Scan rotation complete - advancing to waypoint {self.current_waypoint_idx}')


    # ----- Utilities -----
    def status_loop(self):
        """Periodic status logging."""
        if not self.enabled:
            return

        parts = ["AUTONOMOUS_NAV"]

        if self.current_pose is None:
            parts.append("NO_POSE")
            self.get_logger().info("Status: " + " | ".join(parts))
            return

        # Current position
        curr_x = self.current_pose.pose.position.x
        curr_y = self.current_pose.pose.position.y
        curr_z = self.current_pose.pose.position.z
        curr_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
        parts.append(f"pos=({curr_x:.2f},{curr_y:.2f},{curr_z:.2f})")
        parts.append(f"yaw={math.degrees(curr_yaw):.1f}°")

        # State
        parts.append(f"state={self.waypoint_state}")

        if not self.planned_path:
            parts.append("NO_PATH")
        else:
            # Path information
            parts.append(f"waypoint={self.current_waypoint_idx}/{len(self.planned_path)}")

            # Distance to current waypoint or rotation target
            if self.current_waypoint_idx < len(self.planned_path):
                if self.waypoint_state == 'MOVING':
                    wp = self.planned_path[self.current_waypoint_idx]
                    dx = wp.pose.position.x - curr_x
                    dy = wp.pose.position.y - curr_y
                    dist = math.hypot(dx, dy)
                    is_final = (self.current_waypoint_idx == len(self.planned_path) - 1)
                    tol = self.waypoint_tolerance / 5.0 if is_final else self.waypoint_tolerance
                    parts.append(f"target=({wp.pose.position.x:.2f},{wp.pose.position.y:.2f})")
                    parts.append(f"dist={dist:.2f}m")
                    if is_final:
                        parts.append(f"FINAL_WP(tol={tol:.3f}m)")
                elif self.waypoint_state == 'SCANNING' and self.target_yaw is not None:
                    yaw_error = self._normalize_angle(self.target_yaw - curr_yaw)
                    parts.append(f"target_yaw={math.degrees(self.target_yaw):.1f}°")
                    parts.append(f"yaw_error={math.degrees(yaw_error):.1f}°")
            else:
                parts.append("PATH_COMPLETE")

        self.get_logger().info("Status: " + " | ".join(parts))

    def _publish_zero_velocity(self):
        """Stop the drone."""
        self.cmd_vel_pub.publish(Twist())



def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()