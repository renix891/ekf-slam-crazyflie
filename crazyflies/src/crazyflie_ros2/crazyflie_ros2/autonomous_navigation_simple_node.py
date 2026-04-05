"""
Autonomous Navigation Simple Node for Crazyflie (Test Sequence)

Drop-in replacement for autonomous_navigation_node that executes a test sequence:
1. Rotate 90 degrees
2. Move forward 1 meter
3. Rotate 90 degrees
4. Move backward 1 meter

Subscribes to:
  /crazyflie/pose (geometry_msgs/PoseStamped)

Publishes:
  /cmd_vel (geometry_msgs/Twist)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_srvs.srv import SetBool, Trigger
import math


class AutonomousNavigationSimpleNode(Node):
    """ROS2 node for executing a simple test sequence."""

    def __init__(self):
        super().__init__('autonomous_navigation_node')  # Keep same name for drop-in replacement

        # Parameters (same as original for compatibility)
        self.declare_parameter('waypoint_tolerance', 0.2)
        self.declare_parameter('scanning_yaw_rate', 1.0)  # Max yaw rate limit (rad/s)
        self.declare_parameter('flight_height', 0.3)
        self.declare_parameter('rotation_tolerance', 0.1)  # rad
        self.declare_parameter('speed_factor', 0.30)  # m/s - should match crazyflie_node speed_factor
        self.declare_parameter('yaw_kp', 2.0)  # Proportional gain for yaw control

        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.scanning_yaw_rate = self.get_parameter('scanning_yaw_rate').value
        self.yaw_rate = self.scanning_yaw_rate  # Max yaw rate limit
        self.flight_height = self.get_parameter('flight_height').value
        self.rotation_tolerance = float(self.get_parameter('rotation_tolerance').value)
        self.speed_factor = float(self.get_parameter('speed_factor').value)
        self.yaw_kp = float(self.get_parameter('yaw_kp').value)

        # State
        self.enabled = False
        self.current_pose = None
        self.phase = 'IDLE'  # IDLE -> ROTATE_1 -> MOVE_FORWARD -> ROTATE_2 -> MOVE_BACKWARD -> COMPLETE

        # Movement tracking
        self.target_yaw = None
        self.phase_start_time = None
        self.movement_duration = None

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, '/crazyflie/pose', self.pose_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Services (same as original)
        self.enable_srv = self.create_service(
            SetBool, '/enable_autonomous', self.enable_callback)

        # Service clients
        self.replan_client = self.create_client(Trigger, '/trigger_replanning')

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz

        self.get_logger().info('Autonomous Navigation Simple Node initialized (disabled)')
        self.get_logger().info('Will trigger replanning after each scan rotation')

    # ----- Callbacks -----
    def pose_callback(self, msg):
        was_none = self.current_pose is None
        self.current_pose = msg
        if was_none:
            self.get_logger().info(
                f'Received first pose: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def enable_callback(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f"Autonomous navigation {'enabled' if self.enabled else 'disabled'}"
        self.get_logger().info(response.message)

        if self.enabled:
            # Start the test sequence
            if self.current_pose is None:
                response.success = False
                response.message = "Cannot enable: no pose available"
                self.enabled = False
                self.get_logger().error(response.message)
                return response

            # Start with first rotation (90 degrees from current yaw)
            current_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
            self.target_yaw = self._normalize_angle(current_yaw + math.pi / 2.0)
            self.phase = 'ROTATE_1'
            self.get_logger().info(
                f'Starting test sequence - ROTATE_1: target_yaw={math.degrees(self.target_yaw):.1f}°')
        else:
            self._publish_zero_velocity()
            self.phase = 'IDLE'

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

    def _publish_zero_velocity(self):
        """Stop the drone."""
        self.cmd_vel_pub.publish(Twist())

    # ----- Movement Functions -----
    def move_relative(self, distance: float, direction_unit_vector: list) -> bool:
        """
        Command relative position movement using time-based control.

        Args:
            distance: Distance to travel in meters
            direction_unit_vector: Direction as unit vector [x, y, z] in body frame

        Returns:
            True if movement complete, False otherwise
        """
        if self.current_pose is None:
            return False

        # Initialize movement on first call
        if self.phase_start_time is None:
            # Calculate required time based on speed_factor
            self.movement_duration = distance / self.speed_factor
            self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f'Starting relative movement: {distance:.2f}m at {self.speed_factor:.2f}m/s '
                f'for {self.movement_duration:.2f}s, direction={direction_unit_vector}')

        # Check elapsed time
        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.phase_start_time

        if elapsed >= self.movement_duration:
            self.get_logger().info(
                f'Relative movement complete - elapsed={elapsed:.2f}s')
            self._publish_zero_velocity()
            # Reset for next movement
            self.phase_start_time = None
            self.movement_duration = None
            return True

        # Send velocity command with unit vector
        cmd = Twist()
        cmd.linear.x = float(direction_unit_vector[0])
        cmd.linear.y = float(direction_unit_vector[1]) if len(direction_unit_vector) > 1 else 0.0

        # Altitude hold
        z_err = self.flight_height - self.current_pose.pose.position.z
        cmd.linear.z = max(min(1.0 * z_err, 0.2), -0.2)

        self.cmd_vel_pub.publish(cmd)
        return False

    def rotate_to_target(self, target_yaw: float) -> bool:
        """
        Rotate to target yaw angle using proportional control.
        Yaw rate is proportional to error, clamped to max yaw_rate.

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
        yaw_rate_cmd = max(min(yaw_rate_cmd, self.yaw_rate), -self.yaw_rate)

        # Send rotation command
        cmd = Twist()
        cmd.angular.z = yaw_rate_cmd

        # Altitude hold
        z_err = self.flight_height - self.current_pose.pose.position.z
        cmd.linear.z = max(min(1.0 * z_err, 0.2), -0.2)

        self.cmd_vel_pub.publish(cmd)
        return False

    def move_forward(self, distance: float) -> bool:
        """
        Move forward in body frame for specified distance.
        Uses move_relative with forward unit vector [1.0, 0.0].

        Args:
            distance: Distance to travel in meters

        Returns:
            True if movement complete, False otherwise
        """
        return self.move_relative(distance, [1.0, 0.0])

    def move_backward(self, distance: float) -> bool:
        """
        Move backward in body frame for specified distance.
        Uses move_relative with backward unit vector [-1.0, 0.0].

        Args:
            distance: Distance to travel in meters

        Returns:
            True if movement complete, False otherwise
        """
        return self.move_relative(distance, [-1.0, 0.0])

    # ----- Control Loop -----
    def control_loop(self):
        """Main control loop - executes test sequence when enabled."""
        if not self.enabled or self.current_pose is None:
            return

        if self.phase == 'ROTATE_1':
            # First 90-degree rotation
            if self.rotate_to_target(self.target_yaw):
                # Rotation complete, trigger replanning
                self._trigger_replanning()
                # Then start forward movement
                self.phase = 'MOVE_FORWARD'
                self.get_logger().info('Transitioning to MOVE_FORWARD phase')

        elif self.phase == 'MOVE_FORWARD':
            # Move forward 1 meter
            if self.move_forward(1.0):
                # Forward movement complete, start second rotation
                current_yaw = self._quaternion_to_yaw(self.current_pose.pose.orientation)
                self.target_yaw = self._normalize_angle(current_yaw + math.pi / 2.0)
                self.phase = 'ROTATE_2'
                self.get_logger().info(
                    f'Transitioning to ROTATE_2 phase - target_yaw={math.degrees(self.target_yaw):.1f}°')

        elif self.phase == 'ROTATE_2':
            # Second 90-degree rotation
            if self.rotate_to_target(self.target_yaw):
                # Rotation complete, trigger replanning
                self._trigger_replanning()
                # Then start backward movement
                self.phase = 'MOVE_BACKWARD'
                self.get_logger().info('Transitioning to MOVE_BACKWARD phase')

        elif self.phase == 'MOVE_BACKWARD':
            # Move backward 1 meter
            if self.move_backward(1.0):
                # Backward movement complete, sequence done
                self.phase = 'COMPLETE'
                self.enabled = False
                self.get_logger().info('Test sequence COMPLETE!')

        elif self.phase == 'COMPLETE':
            # Sequence complete, hover in place
            self._publish_zero_velocity()

    def _trigger_replanning(self):
        """Trigger path replanning via service call."""
        if not self.replan_client.service_is_ready():
            self.get_logger().warning('Replanning service not available')
            return

        request = Trigger.Request()
        future = self.replan_client.call_async(request)
        future.add_done_callback(self._replan_response_callback)
        self.get_logger().info('Triggered path replanning after scan')

    def _replan_response_callback(self, future):
        """Handle replanning service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Replanning succeeded: {response.message}')
            else:
                self.get_logger().warning(f'Replanning failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Replanning service call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigationSimpleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
