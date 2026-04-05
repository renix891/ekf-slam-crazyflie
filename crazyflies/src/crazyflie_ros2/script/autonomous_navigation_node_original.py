"""
Autonomous Navigation Node for Crazyflie (Waypoint Following with Active Scanning)

Subscribes to path waypoints from a separate path planning node and follows them
with active scanning. Enabled/disabled via service by mode_manager_node.

Subscribes to:
  /planned_path (nav_msgs/Path) - waypoints from path planner
  /crazyflie/pose (geometry_msgs/PoseStamped)

Publishes:
  /cmd_vel (geometry_msgs/Twist)
  /nav/cmd_vel_plan (geometry_msgs/Twist)
  /nav/cmd_vel_active (geometry_msgs/Twist)
"""

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import SetBool

import numpy as np
import math


class ActiveScanningController:
    """
    Wrapper that converts world-frame planned velocity into body-frame cmd_vel
    while applying a continuous yaw spin for active sensing. Includes a small
    lookahead to compensate for the rotation that will occur before command
    execution takes effect.
    """

    def __init__(self, node: Node):
        self.node = node
        # Parameters
        node.declare_parameter('active_scanning.enabled', True)
        node.declare_parameter('active_scanning.lookahead_tau', 0.3)
        node.declare_parameter('active_scanning.vz_kp', 1.0)
        node.declare_parameter('active_scanning.speed_scale', 1.0)

        self.enabled = bool(node.get_parameter('active_scanning.enabled').value)
        self.lookahead_tau = float(node.get_parameter('active_scanning.lookahead_tau').value)
        self.vz_kp = float(node.get_parameter('active_scanning.vz_kp').value)
        self.speed_scale = float(node.get_parameter('active_scanning.speed_scale').value)

        # Publishers for introspection
        self.plan_pub = node.create_publisher(Twist, '/nav/cmd_vel_plan', 10)
        self.active_pub = node.create_publisher(Twist, '/nav/cmd_vel_active', 10)

    @staticmethod
    def _yaw_from_quat(q):
        w, x, y, z = q.w, q.x, q.y, q.z
        return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    def compute(self, pose_msg: PoseStamped, v_world: np.ndarray, yaw_rate: float, flight_height: float) -> Twist:
        """
        Args:
          pose_msg: current pose (for yaw, z)
          v_world: np.array([vx_w, vy_w, vz_w]) desired in WORLD frame (m/s)
          yaw_rate: desired spin (rad/s), positive CCW
          flight_height: target altitude (m)
        Returns:
          Twist in BODY frame with angular.z set to yaw_rate
        """
        cmd_plan = Twist()
        cmd_plan.linear.x, cmd_plan.linear.y, cmd_plan.linear.z = float(v_world[0]), float(v_world[1]), float(v_world[2])
        cmd_plan.angular.z = float(yaw_rate)
        self.plan_pub.publish(cmd_plan)

        if not self.enabled or pose_msg is None:
            # Fall back to direct world->body using current yaw
            current_yaw = 0.0 if pose_msg is None else self._yaw_from_quat(pose_msg.pose.orientation)
            R = np.array([[ math.cos(-current_yaw), -math.sin(-current_yaw) ],
                          [ math.sin(-current_yaw),  math.cos(-current_yaw) ]])
            v_body_xy = R @ v_world[:2]
        else:
            # Predict yaw a little into the future (lookahead)
            current_yaw = self._yaw_from_quat(pose_msg.pose.orientation)
            yaw_pred = current_yaw + yaw_rate * self.lookahead_tau
            R = np.array([[ math.cos(-yaw_pred), -math.sin(-yaw_pred) ],
                          [ math.sin(-yaw_pred),  math.cos(-yaw_pred) ]])
            v_body_xy = R @ v_world[:2]

        # Scale speed if requested
        v_body_xy = self.speed_scale * v_body_xy

        # Simple altitude hold around flight_height
        z_err = (flight_height - pose_msg.pose.position.z) if pose_msg is not None else 0.0
        vz_cmd = self.vz_kp * z_err  # bounded naturally by CF node's handling

        cmd = Twist()
        cmd.linear.x = float(v_body_xy[0])
        cmd.linear.y = float(v_body_xy[1])
        cmd.linear.z = float(vz_cmd)
        cmd.angular.z = float(yaw_rate)

        self.active_pub.publish(cmd)
        return cmd


class AutonomousNavigationNode(Node):
    """ROS2 node for waypoint following with active scanning."""

    def __init__(self):
        super().__init__('autonomous_navigation_node')

        # Parameters
        self.declare_parameter('waypoint_tolerance', 0.2)
        self.declare_parameter('scanning_yaw_rate', 1.0)
        self.declare_parameter('flight_height', 0.3)

        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.scanning_yaw_rate = self.get_parameter('scanning_yaw_rate').value
        self.flight_height = self.get_parameter('flight_height').value

        # State
        self.enabled = False
        self.current_pose = None
        self.planned_path = []
        self.current_waypoint_idx = 0

        # Controller
        self.controller = ActiveScanningController(self)

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
        return response


    # ----- Control Loop -----
    def control_loop(self):
        """Main control loop - follows waypoints when enabled."""
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
            self.current_waypoint_idx += 1
            return

        # Move towards current waypoint
        v_dir = np.array([dx, dy], dtype=float) / max(dist, 1e-6)
        v_world = np.array([v_dir[0], v_dir[1], 0.0], dtype=float)
        cmd = self.controller.compute(self.current_pose, v_world,
                                      self.scanning_yaw_rate, self.flight_height)
        self.cmd_vel_pub.publish(cmd)


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
        parts.append(f"pos=({curr_x:.2f},{curr_y:.2f},{curr_z:.2f})")

        if not self.planned_path:
            parts.append("NO_PATH")
        else:
            # Path information
            parts.append(f"waypoint={self.current_waypoint_idx}/{len(self.planned_path)}")

            # Distance to current waypoint
            if self.current_waypoint_idx < len(self.planned_path):
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