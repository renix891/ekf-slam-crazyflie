import math, numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

class BoxLandingNode(Node):
    def __init__(self):
        super().__init__('box_landing_node')

        # Params
        self.declare_parameter('enabled', False)
        self.declare_parameter('flight_height', 0.5)
        self.declare_parameter('search_span_m', 0.6)
        self.declare_parameter('search_step_m', 0.15)
        self.declare_parameter('max_descent_rate', 0.08)
        self.declare_parameter('min_hover_height', 0.12)
        self.declare_parameter('min_edge_points', 20)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('box_landing_speed', 1.0)
        self.declare_parameter('search_scale', 1.0)

        self.enabled = bool(self.get_parameter('enabled').value)
        self.flight_height = float(self.get_parameter('flight_height').value)
        self.search_span = float(self.get_parameter('search_span_m').value)
        self.search_step = float(self.get_parameter('search_step_m').value)
        self.vz_max = float(self.get_parameter('max_descent_rate').value)
        self.min_hover = float(self.get_parameter('min_hover_height').value)
        self.min_edge_points = int(self.get_parameter('min_edge_points').value)
        self.goal_tolerance = float(self.get_parameter('goal_tolerance').value)
        self.box_landing_speed = float(self.get_parameter('box_landing_speed').value)
        self.search_scale = float(self.get_parameter('search_scale').value)

        # State
        self.pose = None
        self.pose_history = []  # Track last few poses for velocity calculation
        self.phase = 'IDLE'      # IDLE -> GRID_SEARCH -> NAVIGATE -> WAIT_COMPLETE -> STOP
        self.edge_points = []  # List of (x, y) positions where edges detected
        self.box_centroid = None
        self.wait_start_time = None

        # Grid search state
        self.grid_center = None  # (x, y) position where first edge detected
        self.grid_sequence = []  # List of (vx_unit, vy_unit, duration) velocity commands
        self.grid_start_time = None

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

        self.get_logger().info('box_landing_node ready (disabled by default)')

    def enable_cb(self, req, res):
        self.enabled = bool(req.data)
        res.success = True
        res.message = f'box_landing_node {"enabled" if self.enabled else "disabled"}'
        self.get_logger().info(res.message)
        if self.enabled:
            # Clear state
            self.edge_points.clear()
            self.box_centroid = None
            self.grid_center = None
            self.grid_sequence = []
            self.grid_start_time = None

            # Immediately start grid search using current position as first edge
            if self.pose is not None:
                x = self.pose.pose.position.x
                y = self.pose.pose.position.y
                self.edge_points.append((x, y))
                self.get_logger().info(
                    f'Box landing enabled - using current position as EDGE POINT #1: ({x:.3f}, {y:.3f})')
                self.grid_center = (x, y)
                # self.grid_center = self._apply_direction_offset(self.grid_center, 0.15)  # Comment out to disable offset
                self.grid_sequence = self._generate_grid_sequence()
                self.grid_start_time = self.get_clock().now().nanoseconds / 1e9
                self.phase = 'GRID_SEARCH'
                self._publish_status('grid_searching')
                total_duration = sum(seg[2] for seg in self.grid_sequence)
                self.get_logger().info(
                    f'Starting grid search immediately - center at ({self.grid_center[0]:.3f}, {self.grid_center[1]:.3f}), '
                    f'{len(self.grid_sequence)} movements, total duration={total_duration:.1f}s')
            else:
                self.get_logger().warn('Box landing enabled but no pose available yet - waiting')
                self.phase = 'IDLE'
                self._publish_status('waiting_for_pose')
        else:
            self.phase = 'IDLE'
            self.cmd_pub.publish(Twist())
            # Note: mode_manager handles enabling/disabling autonomous nav
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

    def edge_cb(self, msg: Bool):
        """Handle edge detection events - record position when edge detected during GRID_SEARCH."""
        if not msg.data or not self.enabled or self.pose is None:
            return

        # Only record edges during GRID_SEARCH phase
        if self.phase == 'GRID_SEARCH':
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            self.edge_points.append((x, y))
            self.get_logger().info(
                f'EDGE POINT #{len(self.edge_points)} recorded at ({x:.3f}, {y:.3f})')

            # Check if we have enough points to stop grid search
            if len(self.edge_points) >= self.min_edge_points:
                self.get_logger().info(
                    f'Collected {len(self.edge_points)} edge points (>= {self.min_edge_points}) - stopping grid search')
                self._compute_and_navigate_to_centroid()

    def _generate_grid_sequence(self):
        """
        Generate velocity sequence for two perpendicular raster/lawnmower grid patterns.
        First pattern: horizontal sweeps (left-right), moving down between rows.
        Second pattern: vertical sweeps (up-down), moving right between columns.
        Grid is centered at drone's current position (grid_center), extends ±search_span/2,
        with spacing of search_step.
        Returns list of (vx_unit, vy_unit, duration) tuples.
        """
        if self.grid_center is None:
            self.get_logger().error('Cannot generate grid: grid_center is None')
            return []

        half_span = self.search_span / 2.0
        step = self.search_step

        # Calculate number of rows and columns
        num_cols = int(self.search_span / step) + 1
        num_rows = int(self.search_span / step) + 1

        # Calculate duration for each segment based on step distance and speed
        # duration = distance / speed
        horizontal_duration = step / self.box_landing_speed * self.search_scale
        vertical_duration = step / self.box_landing_speed * self.search_scale

        sequence = []

        # Start at center, move to top-left corner of grid
        # Move up (positive Y)
        move_up = half_span / self.box_landing_speed * self.search_scale
        sequence.append((0.0, 1.0, move_up))

        # Move left (negative X)
        move_left = half_span / self.box_landing_speed * self.search_scale
        sequence.append((-1.0, 0.0, move_left))

        # HORIZONTAL RASTER PATTERN: sweep right/left on each row, move down between rows
        for row in range(num_rows):
            if row % 2 == 0:
                # Even row: move right across entire span
                sequence.append((1.0, 0.0, self.search_span / self.box_landing_speed * self.search_scale))
            else:
                # Odd row: move left across entire span
                sequence.append((-1.0, 0.0, self.search_span / self.box_landing_speed * self.search_scale))

            # Move down to next row (unless it's the last row)
            if row < num_rows - 1:
                sequence.append((0.0, -1.0, vertical_duration))

        # After horizontal pattern, position depends on num_rows
        # If last row index (num_rows-1) is even, we're at bottom-right, need to move left
        # If last row index is odd, we're at bottom-left, already positioned
        if (num_rows - 1) % 2 == 0:
            # Ended at bottom-right, move to bottom-left
            sequence.append((-1.0, 0.0, self.search_span / self.box_landing_speed * self.search_scale))

        # PERPENDICULAR RASTER PATTERN: sweep up/down on each column, move right between columns
        for col in range(num_cols):
            if col % 2 == 0:
                # Even column: move up across entire span
                sequence.append((0.0, 1.0, self.search_span / self.box_landing_speed * self.search_scale))
            else:
                # Odd column: move down across entire span
                sequence.append((0.0, -1.0, self.search_span / self.box_landing_speed * self.search_scale))

            # Move right to next column (unless it's the last column)
            if col < num_cols - 1:
                sequence.append((1.0, 0.0, horizontal_duration))

        return sequence

    def _apply_direction_offset(self, position: tuple, offset_distance: float) -> tuple:
        """
        Offset a position by a distance in the direction the drone was traveling.
        Calculates velocity from recent pose history.
        Returns new (x, y) position offset in the travel direction.
        """
        if self.pose is None or len(self.pose_history) < 2:
            self.get_logger().warn('Cannot apply direction offset: insufficient pose history')
            return position

        # Calculate velocity from pose history (use last few poses for better estimate)
        # Get timestamps and positions
        recent_poses = self.pose_history[-5:] if len(self.pose_history) >= 5 else self.pose_history
        if len(recent_poses) < 2:
            self.get_logger().warn('Cannot apply direction offset: insufficient recent poses')
            return position

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
            self.get_logger().warn('Cannot apply direction offset: zero time interval')
            return position

        # Calculate velocity in world frame
        vx_world = dx_total / dt_total
        vy_world = dy_total / dt_total

        # Normalize direction vector
        magnitude = math.sqrt(vx_world**2 + vy_world**2)
        if magnitude < 1e-6:
            self.get_logger().warn('Cannot apply direction offset: zero velocity')
            return position

        dir_x = vx_world / magnitude
        dir_y = vy_world / magnitude

        # Apply offset
        new_x = position[0] + dir_x * offset_distance
        new_y = position[1] + dir_y * offset_distance

        self.get_logger().info(
            f'Direction offset: original=({position[0]:.3f}, {position[1]:.3f}), '
            f'direction=({dir_x:.3f}, {dir_y:.3f}), '
            f'velocity=({vx_world:.3f}, {vy_world:.3f}) m/s, '
            f'offset={offset_distance:.3f}m, '
            f'new=({new_x:.3f}, {new_y:.3f})')

        return (new_x, new_y)

    def _call_autonomous_nav(self, enable: bool):
        """Enable/disable autonomous navigation node."""
        if not self.autonomous_nav_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Autonomous navigation service not available')
            return False

        request = SetBool.Request()
        request.data = enable
        future = self.autonomous_nav_client.call_async(request)
        # We don't wait for response, just fire and forget
        self.get_logger().info(f'Called autonomous navigation: {"ENABLE" if enable else "DISABLE"}')
        return True

    def _compute_and_navigate_to_centroid(self):
        """Compute box centroid from edge points and navigate to it."""
        if len(self.edge_points) < 3:
            self.get_logger().warn(
                f'Not enough edge points: {len(self.edge_points)} < {self.min_edge_points}')
            return

        # Compute centroid
        xs = [p[0] for p in self.edge_points]
        ys = [p[1] for p in self.edge_points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        self.box_centroid = (cx, cy)

        self.get_logger().info(
            f'Box centroid computed: ({cx:.3f}, {cy:.3f}) from {len(self.edge_points)} edge points')

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
        self.get_logger().info('Transitioning to NAVIGATE phase - autonomous navigation enabled')

    def loop(self):
        if not self.enabled or self.pose is None:
            return

        if self.phase == 'GRID_SEARCH':
            # Execute time-based velocity sequence for grid pattern
            # Edges detected via edge_cb will stop this early if min_edge_points reached
            if self.grid_start_time is None:
                self.get_logger().warn('GRID_SEARCH: No start time set')
                return

            if not self.grid_sequence:
                self.get_logger().error('GRID_SEARCH: No sequence available!')
                self.phase = 'WAIT_COMPLETE'
                self._publish_status('completed')
                self.wait_start_time = self.get_clock().now().nanoseconds / 1e9
                self.cmd_pub.publish(Twist())
                return

            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.grid_start_time
            total_duration = sum(seg[2] for seg in self.grid_sequence)

            # Log progress periodically
            if not hasattr(self, '_grid_log_count'):
                self._grid_log_count = 0
            self._grid_log_count += 1
            if self._grid_log_count % 20 == 0:  # Every 2 seconds
                self.get_logger().info(
                    f'GRID_SEARCH: elapsed={elapsed:.1f}s/{total_duration:.1f}s, edges={len(self.edge_points)}')

            # Check if sequence complete
            if elapsed >= total_duration:
                self.get_logger().info(
                    f'GRID_SEARCH: Sequence complete - {len(self.edge_points)} edges detected')

                # Check if we have enough edges to attempt navigation
                if len(self.edge_points) >= 3:
                    if len(self.edge_points) >= self.min_edge_points:
                        self.get_logger().info(
                            f'Sufficient edges collected ({len(self.edge_points)} >= {self.min_edge_points})')
                    else:
                        self.get_logger().warn(
                            f'Only {len(self.edge_points)} edges collected (target: {self.min_edge_points}), '
                            f'but attempting navigation with available points')
                    self._compute_and_navigate_to_centroid()
                else:
                    self.get_logger().warn(
                        f'Insufficient edges for navigation: {len(self.edge_points)} < 3 (minimum required)')
                    self.phase = 'WAIT_COMPLETE'
                    self._publish_status('completed')
                    self.wait_start_time = self.get_clock().now().nanoseconds / 1e9
                    self.cmd_pub.publish(Twist())
                return

            # Find current segment in sequence
            cumulative_time = 0.0
            for vx_unit, vy_unit, duration in self.grid_sequence:
                if elapsed < cumulative_time + duration:
                    v = Twist()
                    # Apply speed scaling to unit vectors
                    v.linear.x = vx_unit * self.box_landing_speed
                    v.linear.y = vy_unit * self.box_landing_speed
                    # Altitude hold
                    if self.pose:
                        zerr = self.flight_height - self.pose.pose.position.z
                        v.linear.z = max(min(1.0 * zerr, 0.2), -0.2)
                    v.angular.z = 0.0
                    self.cmd_pub.publish(v)
                    return
                cumulative_time += duration

            # Shouldn't reach here, but publish stop just in case
            self.cmd_pub.publish(Twist())
            return

        if self.phase == 'NAVIGATE':
            # Wait for autonomous navigation to reach centroid
            if self.box_centroid is None:
                self.get_logger().error('NAVIGATE: No centroid set!')
                return

            cx, cy = self.box_centroid
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
                    f'NAVIGATE: Reached box centroid! Distance={dist:.3f}m < {self.goal_tolerance}m')

                # Disable autonomous navigation
                self._call_autonomous_nav(False)

                # Transition to WAIT_COMPLETE (hover for 1 second before delegating back)
                self.phase = 'WAIT_COMPLETE'
                self._publish_status('completing')
                self.wait_start_time = self.get_clock().now().nanoseconds / 1e9
                self.cmd_pub.publish(Twist())  # Stop moving
                self.get_logger().info(f'Transitioning to WAIT_COMPLETE - hovering for 1 second')
            return

        if self.phase == 'WAIT_COMPLETE':
            # Hover for 1 second before delegating back to mode_manager
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
    node = BoxLandingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
