import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

class BoxLandingSimpleNode(Node):
    def __init__(self):
        super().__init__('box_landing_simple_node')

        # Params
        self.declare_parameter('enabled', False)
        self.declare_parameter('flight_height', 0.5)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('box_landing_speed', 1.0)
        self.declare_parameter('rotation_speed', 0.5)  # rad/s for yaw rotation
        self.declare_parameter('rotation_tolerance', 0.1)  # rad tolerance for yaw
        self.declare_parameter('second_edge_timeout', 10.0)  # seconds to wait for second edge

        self.enabled = bool(self.get_parameter('enabled').value)
        self.flight_height = float(self.get_parameter('flight_height').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.box_landing_speed = float(self.get_parameter('box_landing_speed').value)
        self.rotation_speed = float(self.get_parameter('rotation_speed').value)
        self.rotation_tolerance = float(self.get_parameter('rotation_tolerance').value)
        self.second_edge_timeout = float(self.get_parameter('second_edge_timeout').value)

        # State
        self.pose = None
        self.pose_history = []  # Track last few poses for velocity calculation
        self.phase = 'IDLE'      # IDLE -> WAIT_FIRST_EDGE -> ROTATE -> FORWARD -> BACKWARD -> WAIT_SECOND_EDGE -> NAVIGATE -> WAIT_COMPLETE -> STOP
        self.edge_points = []  # List of (x, y) positions where edges detected
        self.travel_direction = None  # (vx, vy) direction in world frame
        self.target_yaw = None  # Target yaw angle to face
        self.phase_start_time = None
        self.wait_start_time = None

        # IO
        self.pose_sub = self.create_subscription(PoseStamped, '/crazyflie/pose', self.pose_cb, 10)
        self.edge_event_sub = self.create_subscription(Bool, '/perception/edge_event', self.edge_cb, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.estop_pub = self.create_publisher(Bool, '/crazyflie/emergency_stop', 10)
        self.status_pub = self.create_publisher(String, '/box_landing/status', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Service clients
        self.autonomous_nav_client = self.create_client(SetBool, '/enable_autonomous')

        # Service
        self.enable_srv = self.create_service(SetBool, '/enable_box_landing', self.enable_cb)

        # Timer
        self.timer = self.create_timer(0.1, self.loop)  # 10 Hz

        self.get_logger().info('box_landing_simple_node ready (disabled by default)')

    def enable_cb(self, req, res):
        self.enabled = bool(req.data)
        res.success = True
        res.message = f'box_landing_simple_node {"enabled" if self.enabled else "disabled"}'
        self.get_logger().info(res.message)
        if self.enabled:
            # Preserve the last edge (triggering edge from mode manager)
            # Clear all except the last edge point
            if len(self.edge_points) > 0:
                last_edge = self.edge_points[-1]
                self.edge_points.clear()
                self.edge_points.append(last_edge)
                self.get_logger().info(
                    f'Box landing enabled - using triggering edge at ({last_edge[0]:.3f}, {last_edge[1]:.3f})')
            else:
                self.edge_points.clear()
                self.get_logger().info('Box landing enabled - no triggering edge found')

            # Reset state
            self.travel_direction = None
            self.target_yaw = None
            self.phase_start_time = None

            # If we have a triggering edge, process it immediately
            if len(self.edge_points) > 0:
                # Calculate travel direction
                self.travel_direction = self._calculate_travel_direction()
                if self.travel_direction is None:
                    self.get_logger().error('Cannot calculate travel direction - handing control to mode_manager')
                    self._publish_status('completed')
                    self.phase = 'STOP'
                    self.enabled = False
                    res.success = False
                    res.message = 'Failed to calculate travel direction'
                    return res

                # Calculate target yaw to face travel direction
                vx, vy = self.travel_direction
                self.target_yaw = math.atan2(vy, vx)
                self.get_logger().info(
                    f'Target yaw calculated: {math.degrees(self.target_yaw):.1f} degrees')

                # Start rotating immediately
                self.phase = 'ROTATE'
                self._publish_status('rotating')
                self.get_logger().info('Starting ROTATE phase immediately with triggering edge')
            else:
                # No triggering edge, wait for first edge
                self.phase = 'WAIT_FIRST_EDGE'
                self._publish_status('waiting_first_edge')
                self.get_logger().info('Waiting for first edge detection')
        else:
            self.phase = 'IDLE'
            self.cmd_pub.publish(Twist())
            self.get_logger().info('Box landing disabled - stopping')
        return res

    def _publish_status(self, status: str):
        """Publish status for mode manager."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    def pose_cb(self, msg):
        was_none = self.pose is None

        # Track pose history for velocity calculation (keep last 10 poses)
        if self.pose is not None:
            self.pose_history.append(self.pose)
            if len(self.pose_history) > 10:
                self.pose_history.pop(0)

        self.pose = msg
        if was_none:
            self.get_logger().info(
                f'Received first pose: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def _quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle (rotation around Z-axis)."""
        # yaw = atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))
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

    def _calculate_travel_direction(self):
        """
        Calculate travel direction from pose history.
        Returns (vx, vy) in world frame, or None if insufficient data.
        """
        if self.pose is None or len(self.pose_history) < 2:
            self.get_logger().warn('Cannot calculate travel direction: insufficient pose history')
            return None

        # Calculate velocity from pose history
        recent_poses = self.pose_history[-5:] if len(self.pose_history) >= 5 else self.pose_history
        if len(recent_poses) < 2:
            self.get_logger().warn('Cannot calculate travel direction: insufficient recent poses')
            return None

        # Calculate average velocity from pose differences
        dx_total = 0.0
        dy_total = 0.0
        dt_total = 0.0

        for i in range(len(recent_poses) - 1):
            p1 = recent_poses[i]
            p2 = recent_poses[i + 1]

            # Time difference
            t1 = p1.header.stamp.sec + p1.header.stamp.nanosec / 1e9
            t2 = p2.header.stamp.sec + p2.header.stamp.nanosec / 1e9
            dt = t2 - t1

            if dt > 1e-6:  # Valid time difference
                dx = p2.pose.position.x - p1.pose.position.x
                dy = p2.pose.position.y - p1.pose.position.y
                dx_total += dx
                dy_total += dy
                dt_total += dt

        if dt_total < 1e-6:
            self.get_logger().warn('Cannot calculate travel direction: zero time interval')
            return None

        # Calculate velocity in world frame
        vx_world = dx_total / dt_total
        vy_world = dy_total / dt_total

        # Check if velocity is significant
        magnitude = math.sqrt(vx_world**2 + vy_world**2)
        if magnitude < 1e-3:
            self.get_logger().warn('Cannot calculate travel direction: velocity too small')
            return None

        self.get_logger().info(
            f'Travel direction calculated: vx={vx_world:.3f}, vy={vy_world:.3f}, '
            f'magnitude={magnitude:.3f} m/s')

        return (vx_world, vy_world)

    def edge_cb(self, msg: Bool):
        """Handle edge detection events - records edges even when disabled."""
        if not msg.data or self.pose is None:
            return

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        # Always record edge (even when disabled, so mode manager's triggering edge is captured)
        self.edge_points.append((x, y))

        # Log based on whether we're enabled
        if not self.enabled:
            self.get_logger().info(f'Edge recorded at ({x:.3f}, {y:.3f}) [node disabled]')
            return

        # First edge detection (when enabled and waiting)
        if self.phase == 'WAIT_FIRST_EDGE':
            self.get_logger().info(f'FIRST EDGE detected at ({x:.3f}, {y:.3f})')

            # Calculate travel direction
            self.travel_direction = self._calculate_travel_direction()
            if self.travel_direction is None:
                self.get_logger().error('Cannot calculate travel direction - handing control to mode_manager')
                self._publish_status('completed')
                self.phase = 'STOP'
                self.enabled = False
                return

            # Calculate target yaw to face travel direction
            vx, vy = self.travel_direction
            self.target_yaw = math.atan2(vy, vx)
            self.get_logger().info(
                f'Target yaw calculated: {math.degrees(self.target_yaw):.1f} degrees')

            # Transition to ROTATE phase
            self.phase = 'ROTATE'
            self._publish_status('rotating')
            self.get_logger().info('Transitioning to ROTATE phase')

        # Second edge detection
        elif self.phase == 'WAIT_SECOND_EDGE':
            # Edge already appended at line 217
            self.get_logger().info(f'SECOND EDGE detected at ({x:.3f}, {y:.3f})')

            # Calculate center between two edge points
            if len(self.edge_points) >= 2:
                cx = (self.edge_points[0][0] + self.edge_points[1][0]) / 2.0
                cy = (self.edge_points[0][1] + self.edge_points[1][1]) / 2.0
                self.get_logger().info(
                    f'Centroid calculated: ({cx:.3f}, {cy:.3f}) from 2 edge points')

                # Create and publish goal pose
                goal_msg = PoseStamped()
                goal_msg.header.stamp = self.get_clock().now().to_msg()
                goal_msg.header.frame_id = 'world'
                goal_msg.pose.position.x = cx
                goal_msg.pose.position.y = cy
                goal_msg.pose.position.z = self.flight_height
                goal_msg.pose.orientation.w = 1.0
                self.goal_pub.publish(goal_msg)
                self.get_logger().info(f'Published goal_pose to ({cx:.3f}, {cy:.3f})')

                # Enable autonomous navigation
                self._call_autonomous_nav(True)

                # Transition to NAVIGATE phase
                self.phase = 'NAVIGATE'
                self._publish_status('navigating')
                self.centroid = (cx, cy)
                self.get_logger().info('Transitioning to NAVIGATE phase - autonomous navigation enabled')
            else:
                self.get_logger().error('Cannot calculate center - handing control to mode_manager')
                self._publish_status('completed')
                self.phase = 'STOP'
                self.enabled = False

    def _call_autonomous_nav(self, enable: bool):
        """Enable/disable autonomous navigation node."""
        if not self.autonomous_nav_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Autonomous navigation service not available')
            return False

        request = SetBool.Request()
        request.data = enable
        future = self.autonomous_nav_client.call_async(request)
        self.get_logger().info(f'Called autonomous navigation: {"ENABLE" if enable else "DISABLE"}')
        return True

    def loop(self):
        if not self.enabled or self.pose is None:
            return

        if self.phase == 'ROTATE':
            # Rotate to face travel direction
            if self.target_yaw is None:
                self.get_logger().error('ROTATE: No target yaw set!')
                return

            # Get current yaw
            current_yaw = self._quaternion_to_yaw(self.pose.pose.orientation)

            # Calculate yaw error
            yaw_error = self._normalize_angle(self.target_yaw - current_yaw)

            # Check if rotation complete
            if abs(yaw_error) < self.rotation_tolerance:
                self.get_logger().info(
                    f'ROTATE: Rotation complete - error={math.degrees(yaw_error):.1f} deg')

                # Transition to FORWARD phase
                self.phase = 'FORWARD'
                self._publish_status('moving_forward')
                self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info('Transitioning to FORWARD phase')
                return

            # Send rotation command
            v = Twist()
            v.angular.z = self.rotation_speed if yaw_error > 0 else -self.rotation_speed
            # Altitude hold
            zerr = self.flight_height - self.pose.pose.position.z
            v.linear.z = max(min(1.0 * zerr, 0.2), -0.2)
            self.cmd_pub.publish(v)
            return

        if self.phase == 'FORWARD':
            # Move forward for 2 seconds
            if self.phase_start_time is None:
                self.get_logger().error('FORWARD: No start time set!')
                return

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.phase_start_time

            if elapsed >= 2.0:
                self.get_logger().info('FORWARD: 2 seconds elapsed')

                # Transition to BACKWARD phase
                self.phase = 'BACKWARD'
                self._publish_status('moving_backward')
                self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info('Transitioning to BACKWARD phase')
                return

            # Send forward command (in body frame)
            v = Twist()
            v.linear.x = self.box_landing_speed  # Forward in body frame
            # Altitude hold
            zerr = self.flight_height - self.pose.pose.position.z
            v.linear.z = max(min(1.0 * zerr, 0.2), -0.2)
            self.cmd_pub.publish(v)
            return

        if self.phase == 'BACKWARD':
            # Move backward for 2 seconds
            if self.phase_start_time is None:
                self.get_logger().error('BACKWARD: No start time set!')
                return

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.phase_start_time

            if elapsed >= 2.0:
                self.get_logger().info('BACKWARD: 2 seconds elapsed')

                # Stop and transition to WAIT_SECOND_EDGE phase
                self.cmd_pub.publish(Twist())
                self.phase = 'WAIT_SECOND_EDGE'
                self._publish_status('waiting_second_edge')
                self.phase_start_time = self.get_clock().now().nanoseconds / 1e9
                self.get_logger().info('Transitioning to WAIT_SECOND_EDGE phase')
                return

            # Send backward command (in body frame)
            v = Twist()
            v.linear.x = -self.box_landing_speed  # Backward in body frame
            # Altitude hold
            zerr = self.flight_height - self.pose.pose.position.z
            v.linear.z = max(min(1.0 * zerr, 0.2), -0.2)
            self.cmd_pub.publish(v)
            return

        if self.phase == 'WAIT_SECOND_EDGE':
            # Wait for second edge detection with timeout
            if self.phase_start_time is None:
                self.get_logger().error('WAIT_SECOND_EDGE: No start time set!')
                return

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.phase_start_time

            # Hover while waiting
            v = Twist()
            zerr = self.flight_height - self.pose.pose.position.z
            v.linear.z = max(min(1.0 * zerr, 0.2), -0.2)
            self.cmd_pub.publish(v)

            # Check timeout
            if elapsed >= self.second_edge_timeout:
                self.get_logger().warn(
                    f'WAIT_SECOND_EDGE: Timeout after {self.second_edge_timeout}s - handing control to mode_manager')
                self._publish_status('completed')
                self.phase = 'STOP'
                self.enabled = False
            return

        if self.phase == 'NAVIGATE':
            # Wait for autonomous navigation to reach centroid
            if not hasattr(self, 'centroid') or self.centroid is None:
                self.get_logger().error('NAVIGATE: No centroid set!')
                return

            cx, cy = self.centroid
            px = self.pose.pose.position.x
            py = self.pose.pose.position.y
            dist = math.hypot(cx - px, cy - py)

            # Log progress periodically
            if not hasattr(self, '_nav_log_count'):
                self._nav_log_count = 0
            self._nav_log_count += 1
            if self._nav_log_count % 20 == 0:  # Every 2 seconds
                self.get_logger().info(
                    f'NAVIGATE: Current ({px:.3f}, {py:.3f}), '
                    f'Target ({cx:.3f}, {cy:.3f}), distance={dist:.3f}m')

            # Check if reached goal
            if dist < self.goal_tolerance:
                self.get_logger().info(
                    f'NAVIGATE: Reached centroid! Distance={dist:.3f}m < {self.goal_tolerance}m')

                # Disable autonomous navigation
                self._call_autonomous_nav(False)

                # Transition to WAIT_COMPLETE
                self.phase = 'WAIT_COMPLETE'
                self._publish_status('completing')
                self.wait_start_time = self.get_clock().now().nanoseconds / 1e9
                self.cmd_pub.publish(Twist())
                self.get_logger().info('Transitioning to WAIT_COMPLETE - hovering for 1 second')
            return

        if self.phase == 'WAIT_COMPLETE':
            # Hover for 1 second before delegating back
            if self.wait_start_time is None:
                self.get_logger().warn('WAIT_COMPLETE: No wait start time set')
                return

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.wait_start_time

            # Continue hovering
            self.cmd_pub.publish(Twist())

            # Check if wait period complete
            if elapsed >= 1.0:
                self.get_logger().info('WAIT_COMPLETE: 1 second elapsed - delegating back to mode_manager')
                self._publish_status('completed')
                self.phase = 'STOP'
                self.enabled = False
                self.get_logger().info('Box landing sequence complete!')
            return

        if self.phase == 'STOP':
            # Complete - do nothing
            return


def main():
    rclpy.init()
    node = BoxLandingSimpleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
