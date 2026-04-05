"""
Mode Manager Node for Crazyflie Mission Control

Manages the finite state machine for the entire mission, coordinating
different specialized nodes by enabling/disabling them as needed.

States:
  - IDLE: Waiting for goal
  - TAKEOFF: Altitude acquisition
  - NAVIGATION: Path following to goal
  - BOX_LANDING: Precision landing on detected box
  - LAND: Emergency landing sequence
  - WAITING: Post-landing, waiting for REFLY command

Subscribes to:
  /crazyflie/pose (geometry_msgs/PoseStamped)
  /goal_pose (geometry_msgs/PoseStamped)
  /perception/edge_event (std_msgs/Bool)
  /box_landing/status (std_msgs/String)

Service Clients:
  /enable_autonomous (std_srvs/SetBool) - controls autonomous_navigation_node
  /enable_box_landing (std_srvs/SetBool) - controls box_landing_node

Service Servers:
  /refly (std_srvs/Trigger) - returns to IDLE state for new mission

Publishes:
  /crazyflie/emergency_stop (std_msgs/Bool)
  /mode_manager/state (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger


class ModeManagerNode(Node):
    """Central FSM for mission control."""

    def __init__(self):
        super().__init__('mode_manager_node')

        # Parameters
        self.declare_parameter('flight_height', 0.5)
        self.declare_parameter('altitude_tolerance', 0.05)
        self.declare_parameter('edge_detection_delay', 5.0)
        self.declare_parameter('landing_descent_rate', 0.01)  # m per control cycle
        self.declare_parameter('min_landing_height', 0.12)

        self.flight_height = self.get_parameter('flight_height').value
        self.altitude_tolerance = self.get_parameter('altitude_tolerance').value
        self.edge_detection_delay = self.get_parameter('edge_detection_delay').value
        self.landing_descent_rate = self.get_parameter('landing_descent_rate').value
        self.min_landing_height = self.get_parameter('min_landing_height').value

        # State
        self.state = 'IDLE'
        self.current_pose = None
        self.goal_pose = None
        self.navigation_start_time = None
        self.current_landing_height = self.flight_height

        # Track node enabled states to avoid redundant service calls
        self.autonomous_nav_enabled = False
        self.box_landing_enabled = False

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/crazyflie/pose', self.pose_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.edge_event_sub = self.create_subscription(
            Bool, '/perception/edge_event', self.edge_callback, 10)
        self.box_landing_status_sub = self.create_subscription(
            String, '/box_landing/status', self.box_landing_status_callback, 10)

        # Publishers
        self.emergency_stop_pub = self.create_publisher(Bool, '/crazyflie/emergency_stop', 10)
        self.state_pub = self.create_publisher(String, '/mode_manager/state', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Service clients
        self.autonomous_nav_client = self.create_client(SetBool, '/enable_autonomous')
        self.box_landing_client = self.create_client(SetBool, '/enable_box_landing')

        # Service servers
        self.refly_service = self.create_service(Trigger, '/refly', self.refly_callback)

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.status_timer = self.create_timer(5.0, self.status_loop)

        self.get_logger().info('Mode Manager Node initialized')
        self.get_logger().info(f'Initial state: {self.state}')

    # ----- Callbacks -----
    def pose_callback(self, msg: PoseStamped):
        was_none = self.current_pose is None
        self.current_pose = msg
        if was_none:
            self.get_logger().info(
                f'Received first pose: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose = msg
        self.get_logger().info(
            f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

    def edge_callback(self, msg: Bool):
        """Handle edge detection events."""
        if not msg.data:
            return

        # Only respond to edge events in NAVIGATION state
        if self.state == 'NAVIGATION':
            if self.navigation_start_time is None:
                return

            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.navigation_start_time

            if elapsed < self.edge_detection_delay:
                self.get_logger().info(
                    f'Edge detected but delay not met ({elapsed:.1f}s < {self.edge_detection_delay}s)')
                return

            # Edge detected - transition to box landing
            self.get_logger().info('EDGE detected → STATE CHANGE: NAVIGATION -> BOX_LANDING')
            self._change_state('BOX_LANDING')

    def box_landing_status_callback(self, msg: String):
        """Handle status updates from box landing node."""
        if msg.data == 'completed' and self.state == 'BOX_LANDING':
            self.get_logger().info('Box landing completed → STATE CHANGE: BOX_LANDING -> LAND')
            self._change_state('LAND')

    def refly_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Handle REFLY service call to restart mission."""
        if self.state == 'WAITING':
            self.get_logger().info('REFLY service called → STATE CHANGE: WAITING -> IDLE')
            self._change_state('IDLE')
            response.success = True
            response.message = 'Transitioning to IDLE state'
        else:
            self.get_logger().warn(f'REFLY service called in {self.state} state, expected WAITING')
            response.success = False
            response.message = f'Cannot REFLY from {self.state} state'
        return response

    # ----- State Management -----
    def _change_state(self, new_state: str):
        """Centralized state change handler."""
        old_state = self.state
        # Proactively zero velocity when leaving a controller-owned state
        if old_state in ('NAVIGATION', 'BOX_LANDING'):
            self._publish_zero_velocity()
        self.state = new_state

        # Publish state change
        state_msg = String()
        state_msg.data = new_state
        self.state_pub.publish(state_msg)

        # Handle state transitions - only enable/disable specific nodes as needed
        if new_state == 'TAKEOFF':
            # Only disable all nodes at start of flight (from IDLE)
            if old_state == 'IDLE':
                self._disable_all_control_nodes()
            # Takeoff is handled directly in control_loop via cmd_vel

        elif new_state == 'NAVIGATION':
            # Disable box_landing if coming from BOX_LANDING
            if old_state == 'BOX_LANDING':
                self._enable_box_landing(False)
            # Enable autonomous navigation
            self._enable_autonomous_navigation(True)
            self.navigation_start_time = self.get_clock().now().nanoseconds / 1e9

        elif new_state == 'BOX_LANDING':
            # Disable autonomous navigation if coming from NAVIGATION
            if old_state == 'NAVIGATION':
                self._enable_autonomous_navigation(False)
            # Enable box landing
            self._enable_box_landing(True)

        elif new_state == 'LAND':
            # Disable whichever node was active
            if old_state == 'NAVIGATION':
                self._enable_autonomous_navigation(False)
            elif old_state == 'BOX_LANDING':
                self._enable_box_landing(False)
            self.current_landing_height = self.flight_height
            # Landing is handled directly in control_loop via cmd_vel

        elif new_state == 'IDLE':
            # Disable whichever node was active
            if old_state == 'NAVIGATION':
                self._enable_autonomous_navigation(False)
            elif old_state == 'BOX_LANDING':
                self._enable_box_landing(False)
            self._publish_zero_velocity()

        elif new_state == 'WAITING':
            # All nodes should already be disabled
            # Just sit and wait for REFLY service call
            pass

        self.get_logger().info(f'STATE CHANGE: {old_state} -> {new_state}')

    def _enable_autonomous_navigation(self, enable: bool):
        """Enable/disable autonomous navigation node."""
        # Check if already in desired state
        if self.autonomous_nav_enabled == enable:
            self.get_logger().debug(
                f'Autonomous navigation already {"enabled" if enable else "disabled"} - skipping')
            return

        if not self.autonomous_nav_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Autonomous navigation service not available')
            return

        request = SetBool.Request()
        request.data = enable
        future = self.autonomous_nav_client.call_async(request)
        self.autonomous_nav_enabled = enable
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'Autonomous navigation {"enabled" if enable else "disabled"}'))

    def _enable_box_landing(self, enable: bool):
        """Enable/disable box landing node."""
        # Check if already in desired state
        if self.box_landing_enabled == enable:
            self.get_logger().debug(
                f'Box landing already {"enabled" if enable else "disabled"} - skipping')
            return

        if not self.box_landing_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Box landing service not available')
            return

        request = SetBool.Request()
        request.data = enable
        future = self.box_landing_client.call_async(request)
        self.box_landing_enabled = enable
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'Box landing {"enabled" if enable else "disabled"}'))

    def _disable_all_control_nodes(self):
        """Disable all control nodes."""
        self._enable_autonomous_navigation(False)
        self._enable_box_landing(False)

    # ----- Control Loop -----
    def control_loop(self):
        """Main control loop - handles state transitions and direct control states."""
        if self.current_pose is None:
            return

        # State machine logic
        if self.state == 'IDLE':
            if self.goal_pose is not None:
                self._change_state('TAKEOFF')
            return

        elif self.state == 'TAKEOFF':
            # Check if altitude reached
            current_alt = self.current_pose.pose.position.z
            alt_error = abs(current_alt - self.flight_height)

            # Log altitude status periodically
            if not hasattr(self, '_takeoff_log_count'):
                self._takeoff_log_count = 0
            self._takeoff_log_count += 1
            if self._takeoff_log_count >= 50:  # Every 5 seconds at 10 Hz
                self.get_logger().info(
                    f'TAKEOFF: current_alt={current_alt:.3f}m, '
                    f'target={self.flight_height:.3f}m, error={alt_error:.3f}m')
                self._takeoff_log_count = 0

            if alt_error < self.altitude_tolerance:
                self.get_logger().info(
                    f'Altitude reached: {current_alt:.3f}m → STATE CHANGE: TAKEOFF -> NAVIGATION')
                self._change_state('NAVIGATION')
            else:
                # Send climb command
                cmd = Twist()
                # Simple altitude control - let the flight controller handle details
                cmd.linear.z = 0.2 if current_alt < self.flight_height else -0.2
                self.cmd_vel_pub.publish(cmd)
            return

        elif self.state == 'NAVIGATION':
            # Navigation is handled by autonomous_navigation_node when enabled
            # This state just monitors for edge detection (handled in edge_callback)
            return

        elif self.state == 'BOX_LANDING':
            # Box landing is handled by box_landing_node when enabled
            # This state just monitors for completion (handled in box_landing_status_callback)
            return

        elif self.state == 'LAND':
            # Gradual descent
            if not hasattr(self, '_last_land_log') or \
               (self.get_clock().now().nanoseconds - self._last_land_log) > 5e8:
                self.get_logger().info(
                    f'LAND: z={self.current_pose.pose.position.z:.2f}m '
                    f'→ target={self.current_landing_height:.2f}m')
                self._last_land_log = self.get_clock().now().nanoseconds

            # Lower target altitude
            self.current_landing_height = max(
                self.min_landing_height,
                self.current_landing_height - self.landing_descent_rate)

            # Send descent command
            cmd = Twist()
            z_error = self.current_landing_height - self.current_pose.pose.position.z
            cmd.linear.z = 1.0 * z_error  # Simple proportional control
            self.cmd_vel_pub.publish(cmd)

            # Check if landing complete
            if self.current_landing_height <= self.min_landing_height or \
               self.current_pose.pose.position.z < 0.15:
                self.get_logger().info('Landing complete - sending emergency stop')
                self._publish_zero_velocity()

                # Send emergency stop
                stop_msg = Bool()
                stop_msg.data = True
                self.emergency_stop_pub.publish(stop_msg)

                self._change_state('WAITING')
            return

        elif self.state == 'WAITING':
            # Waiting for REFLY service call
            # Do nothing, just maintain current state
            return

    def status_loop(self):
        """Periodic status logging."""
        parts = [f"STATE={self.state}"]

        if self.current_pose is None:
            parts.append("NO_POSE")
        else:
            parts.append(f"pose=({self.current_pose.pose.position.x:.1f},"
                        f"{self.current_pose.pose.position.y:.1f},"
                        f"{self.current_pose.pose.position.z:.1f})")

        if self.goal_pose is None:
            parts.append("NO_GOAL")
        else:
            parts.append(f"goal=({self.goal_pose.pose.position.x:.1f},"
                        f"{self.goal_pose.pose.position.y:.1f})")

        self.get_logger().info("Status: " + " | ".join(parts))

    def _publish_zero_velocity(self):
        """Stop the drone."""
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
