#!/usr/bin/env python3
"""
Mission Manager Node for Crazyflie Mission Control

Manages the finite state machine for the entire mission, integrating
box landing logic directly without separate node coordination.

States:
  - IDLE: Startup state
  - TAKEOFF: Using controller takeoff service
  - READY: Hovering, waiting for goal
  - NAVIGATION: Path following to goal
  - BOX_LANDING: Precision landing on detected box
    - GRID_SEARCH: Sequential waypoint grid search
    - NAVIGATE_TO_CENTROID: Move to computed box center
    - WAIT_COMPLETE: Brief hover before completion
  - LAND: Using controller land service
  - WAITING: Post-landing, mission complete

Subscribes to:
  /odom (nav_msgs/Odometry)
  /goal_pose (geometry_msgs/PoseStamped)
  /perception/edge_event (std_msgs/Bool)
  /position_target_reached (std_msgs/Bool)
  /crazyflie/state (std_msgs/Int8)

Service Clients:
  /takeoff (std_srvs/Trigger) - controller takeoff
  /land (std_srvs/Trigger) - controller landing

Publishes:
  /cmd_pos (geometry_msgs/PointStamped)
  /mode_manager/state (std_msgs/String)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String, Int8
from std_srvs.srv import Trigger
import math


class MissionManagerNode(Node):
    """Central FSM for mission control with integrated box landing."""

    def __init__(self):
        super().__init__('mission_manager_node')

        # Parameters
        self.declare_parameter('flight_height', 0.5)
        self.declare_parameter('search_span_m', 0.6)
        self.declare_parameter('search_step_m', 0.15)
        self.declare_parameter('min_edge_points', 20)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('edge_detection_delay', 5.0)

        self.flight_height = self.get_parameter('flight_height').value
        self.search_span = self.get_parameter('search_span_m').value
        self.search_step = self.get_parameter('search_step_m').value
        self.min_edge_points = self.get_parameter('min_edge_points').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.edge_delay = self.get_parameter('edge_detection_delay').value

        # State
        self.state = 'IDLE'
        self.box_landing_phase = None  # GRID_SEARCH, NAVIGATE_TO_CENTROID, WAIT_COMPLETE
        self.current_pose = None
        self.goal_pose = None
        self.navigation_start_time = None

        # Crazyflie controller state (0=FLYING, 1=LANDING, 2=LANDED, 3=TAKING_OFF)
        self.crazyflie_state = None

        # Box landing state
        self.edge_points = []
        self.box_centroid = None
        self.grid_waypoints = []
        self.current_grid_idx = 0
        self.waiting_for_position = False
        self.wait_complete_start = None

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.edge_event_sub = self.create_subscription(
            Bool, '/perception/edge_event', self.edge_callback, 10)
        self.position_reached_sub = self.create_subscription(
            Bool, '/position_target_reached', self.position_reached_callback, 10)
        self.crazyflie_state_sub = self.create_subscription(
            Int8, '/crazyflie/state', self.crazyflie_state_callback, 10)

        # Publishers
        self.state_pub = self.create_publisher(String, '/mode_manager/state', 10)
        self.cmd_pos_pub = self.create_publisher(PointStamped, '/cmd_pos', 10)

        # Service clients
        self.takeoff_client = self.create_client(Trigger, '/takeoff')
        self.land_client = self.create_client(Trigger, '/land')

        # Timers
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10 Hz
        self.status_timer = self.create_timer(5.0, self.status_loop)

        self.get_logger().info('Mission Manager Node initialized')
        self.get_logger().info(f'Initial state: {self.state}')

    # ----- Callbacks -----
    def crazyflie_state_callback(self, msg: Int8):
        """Handle crazyflie controller state updates."""
        was_none = self.crazyflie_state is None
        self.crazyflie_state = msg.data

        state_names = {-1: 'UNINITIALIZED', 0: 'FLYING', 1: 'LANDING', 2: 'LANDED', 3: 'TAKING_OFF'}
        state_name = state_names.get(msg.data, f'UNKNOWN({msg.data})')

        if was_none:
            self.get_logger().info(f'Received first crazyflie state: {state_name}')

    def odom_callback(self, msg: Odometry):
        """Handle odometry updates from controller."""
        was_none = self.current_pose is None
        # Store as PoseStamped for compatibility with rest of code
        pose_msg = PoseStamped()
        pose_msg.header = msg.header
        pose_msg.pose = msg.pose.pose
        self.current_pose = pose_msg
        if was_none:
            self.get_logger().info(
                f'Received first pose: ({msg.pose.pose.position.x:.2f}, '
                f'{msg.pose.pose.position.y:.2f}, {msg.pose.pose.position.z:.2f})')

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose = msg
        self.get_logger().info(
            f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')

        # Transition to NAVIGATION if in READY state
        if self.state == 'READY':
            self.get_logger().info('Goal received → STATE CHANGE: READY -> NAVIGATION')
            self._change_state('NAVIGATION')

    def edge_callback(self, msg: Bool):
        """Handle edge detection events."""
        if not msg.data:
            return

        # Only respond to edge events in NAVIGATION state or BOX_LANDING GRID_SEARCH
        if self.state == 'NAVIGATION':
            if self.navigation_start_time is None:
                return

            current_time = self.get_clock().now().nanoseconds / 1e9
            elapsed = current_time - self.navigation_start_time

            if elapsed < self.edge_delay:
                self.get_logger().info(
                    f'Edge detected but delay not met ({elapsed:.1f}s < {self.edge_delay}s)')
                return

            # Edge detected - transition to box landing
            self.get_logger().info('EDGE detected → STATE CHANGE: NAVIGATION -> BOX_LANDING')
            self._change_state('BOX_LANDING')

        elif self.state == 'BOX_LANDING' and self.box_landing_phase == 'GRID_SEARCH':
            # Record edge point during grid search
            if self.current_pose is not None:
                x = self.current_pose.pose.position.x
                y = self.current_pose.pose.position.y
                self.edge_points.append((x, y))
                self.get_logger().info(
                    f'EDGE POINT #{len(self.edge_points)} recorded at ({x:.3f}, {y:.3f})')

                # Check if we have enough points
                if len(self.edge_points) >= self.min_edge_points:
                    self.get_logger().info(
                        f'Collected {len(self.edge_points)} edge points (>= {self.min_edge_points})')
                    self._compute_and_navigate_to_centroid()

    def position_reached_callback(self, msg: Bool):
        """Handle position target reached notifications from controller."""
        if msg.data and self.waiting_for_position:
            self.waiting_for_position = False
            self.get_logger().debug('Position target reached')

    # ----- State Management -----
    def _change_state(self, new_state: str):
        """Centralized state change handler."""
        old_state = self.state
        self.state = new_state

        # Publish state change
        state_msg = String()
        state_msg.data = new_state
        self.state_pub.publish(state_msg)

        # Handle state transitions
        if new_state == 'TAKEOFF':
            # Call takeoff service
            self._call_takeoff_service()

        elif new_state == 'NAVIGATION':
            self.navigation_start_time = self.get_clock().now().nanoseconds / 1e9

        elif new_state == 'BOX_LANDING':
            # Initialize box landing
            self.box_landing_phase = 'GRID_SEARCH'
            self.edge_points.clear()
            self.box_centroid = None

            # Generate grid waypoints centered at current position
            if self.current_pose is not None:
                cx = self.current_pose.pose.position.x
                cy = self.current_pose.pose.position.y
                self.edge_points.append((cx, cy))
                self.get_logger().info(
                    f'Box landing started - using current position as EDGE POINT #1: ({cx:.3f}, {cy:.3f})')

                self.grid_waypoints = self._generate_grid_waypoints(cx, cy)
                self.current_grid_idx = 0
                self.get_logger().info(
                    f'Generated {len(self.grid_waypoints)} grid waypoints for search')

        elif new_state == 'LAND':
            # Call land service
            self._call_land_service()

        elif new_state == 'IDLE':
            # Reset mission state
            self.goal_pose = None
            self.navigation_start_time = None

        self.get_logger().info(f'STATE CHANGE: {old_state} -> {new_state}')

    def _call_takeoff_service(self):
        """Call controller takeoff service."""
        if not self.takeoff_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Takeoff service not available')
            return

        request = Trigger.Request()
        future = self.takeoff_client.call_async(request)
        future.add_done_callback(self._takeoff_response_callback)
        self.get_logger().info('Calling takeoff service...')

    def _takeoff_response_callback(self, future):
        """Handle takeoff service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Takeoff successful: {response.message}')
                self._change_state('READY')
            else:
                self.get_logger().error(f'Takeoff failed: {response.message}')
                self._change_state('IDLE')
        except Exception as e:
            self.get_logger().error(f'Takeoff service call failed: {e}')
            self._change_state('IDLE')

    def _call_land_service(self):
        """Call controller land service."""
        if not self.land_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Land service not available')
            return

        request = Trigger.Request()
        future = self.land_client.call_async(request)
        future.add_done_callback(self._land_response_callback)
        self.get_logger().info('Calling land service...')

    def _land_response_callback(self, future):
        """Handle land service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Landing successful: {response.message}')
                self._change_state('WAITING')
            else:
                self.get_logger().error(f'Landing failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Land service call failed: {e}')

    # ----- Grid Search -----
    def _generate_grid_waypoints(self, center_x: float, center_y: float):
        """
        Generate waypoint list for two perpendicular raster/lawnmower grid patterns.
        First pattern: horizontal sweeps (left-right), moving down between rows.
        Second pattern: vertical sweeps (up-down), moving right between columns.
        Returns list of (x, y) positions.
        """
        half_span = self.search_span / 2.0
        step = self.search_step

        num_cols = int(self.search_span / step) + 1
        num_rows = int(self.search_span / step) + 1

        waypoints = []

        # Start at top-left corner
        start_x = center_x - half_span
        start_y = center_y + half_span
        waypoints.append((start_x, start_y))

        # HORIZONTAL RASTER PATTERN
        for row in range(num_rows):
            y = start_y - row * step
            if row % 2 == 0:
                # Even row: sweep right
                for col in range(1, num_cols + 1):
                    x = start_x + col * step
                    waypoints.append((x, y))
            else:
                # Odd row: sweep left
                for col in range(num_cols - 1, -1, -1):
                    x = start_x + col * step
                    waypoints.append((x, y))

        # Position after horizontal pattern
        last_x = waypoints[-1][0]
        last_y = waypoints[-1][1]

        # Move to bottom-left if needed
        if last_x != start_x:
            waypoints.append((start_x, last_y))

        # VERTICAL RASTER PATTERN
        for col in range(num_cols):
            x = start_x + col * step
            if col % 2 == 0:
                # Even column: sweep up
                for row in range(num_rows - 1, -1, -1):
                    y = start_y - row * step
                    waypoints.append((x, y))
            else:
                # Odd column: sweep down
                for row in range(num_rows):
                    y = start_y - row * step
                    waypoints.append((x, y))

        return waypoints

    def _compute_and_navigate_to_centroid(self):
        """Compute box centroid from edge points and navigate to it."""
        if len(self.edge_points) < 3:
            self.get_logger().warn(
                f'Not enough edge points: {len(self.edge_points)} < 3')
            return

        # Compute centroid
        xs = [p[0] for p in self.edge_points]
        ys = [p[1] for p in self.edge_points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        self.box_centroid = (cx, cy)

        self.get_logger().info(
            f'Box centroid computed: ({cx:.3f}, {cy:.3f}) from {len(self.edge_points)} edge points')

        # Transition to navigate to centroid phase
        self.box_landing_phase = 'NAVIGATE_TO_CENTROID'
        self._send_position_target(cx, cy, self.flight_height)
        self.waiting_for_position = True

    def _send_position_target(self, x: float, y: float, z: float):
        """Send position target to controller via cmd_pos."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.point.x = x
        msg.point.y = y
        msg.point.z = z
        self.cmd_pos_pub.publish(msg)
        self.get_logger().debug(f'Sent position target: ({x:.3f}, {y:.3f}, {z:.3f})')

    # ----- Control Loop -----
    def control_loop(self):
        """Main control loop - handles state transitions and position commands."""
        # State machine logic
        if self.state == 'IDLE':
            # Auto-takeoff when crazyflie is in LANDED state (2)
            if self.crazyflie_state == 2:  # STATE_LANDED
                self.get_logger().info('Crazyflie is LANDED and ready - initiating takeoff')
                self._change_state('TAKEOFF')
            return

        elif self.state == 'TAKEOFF':
            # Waiting for takeoff service to complete
            return

        elif self.state == 'READY':
            # Hovering, waiting for goal_pose
            # Transition happens in goal_callback when goal is received
            # Need pose data for this state
            if self.current_pose is None:
                return
            return

        elif self.state == 'NAVIGATION':
            # Need pose data for this state
            if self.current_pose is None:
                return
            # Send goal position to controller
            if self.goal_pose is not None and not self.waiting_for_position:
                self._send_position_target(
                    self.goal_pose.pose.position.x,
                    self.goal_pose.pose.position.y,
                    self.flight_height)
                self.waiting_for_position = True
            return

        elif self.state == 'BOX_LANDING':
            # Need pose data for this state
            if self.current_pose is None:
                return
            if self.box_landing_phase == 'GRID_SEARCH':
                # Send next grid waypoint
                if not self.waiting_for_position and self.current_grid_idx < len(self.grid_waypoints):
                    wx, wy = self.grid_waypoints[self.current_grid_idx]
                    self._send_position_target(wx, wy, self.flight_height)
                    self.waiting_for_position = True
                    self.current_grid_idx += 1

                elif self.current_grid_idx >= len(self.grid_waypoints):
                    # Grid search complete
                    self.get_logger().info(
                        f'Grid search complete - {len(self.edge_points)} edges detected')

                    if len(self.edge_points) >= 3:
                        if len(self.edge_points) >= self.min_edge_points:
                            self.get_logger().info(
                                f'Sufficient edges collected ({len(self.edge_points)} >= {self.min_edge_points})')
                        else:
                            self.get_logger().warn(
                                f'Only {len(self.edge_points)} edges collected (target: {self.min_edge_points}), '
                                f'attempting navigation with available points')
                        self._compute_and_navigate_to_centroid()
                    else:
                        self.get_logger().warn(
                            f'Insufficient edges for navigation: {len(self.edge_points)} < 3 (minimum required)')
                        self._change_state('LAND')

            elif self.box_landing_phase == 'NAVIGATE_TO_CENTROID':
                # Check if reached centroid
                if not self.waiting_for_position and self.box_centroid is not None:
                    cx, cy = self.box_centroid
                    px = self.current_pose.pose.position.x
                    py = self.current_pose.pose.position.y
                    dist = math.hypot(cx - px, cy - py)

                    if dist < self.goal_tolerance:
                        self.get_logger().info(
                            f'Reached box centroid! Distance={dist:.3f}m < {self.goal_tolerance}m')
                        self.box_landing_phase = 'WAIT_COMPLETE'
                        self.wait_complete_start = self.get_clock().now().nanoseconds / 1e9

            elif self.box_landing_phase == 'WAIT_COMPLETE':
                # Wait 1 second before transitioning to land
                if self.wait_complete_start is not None:
                    elapsed = self.get_clock().now().nanoseconds / 1e9 - self.wait_complete_start
                    if elapsed >= 1.0:
                        self.get_logger().info('Box landing complete - initiating landing')
                        self._change_state('LAND')
            return

        elif self.state == 'LAND':
            # Waiting for land service to complete
            return

        elif self.state == 'WAITING':
            # Mission complete
            return

    def status_loop(self):
        """Periodic status logging."""
        parts = [f"STATE={self.state}"]

        if self.state == 'BOX_LANDING' and self.box_landing_phase:
            parts.append(f"PHASE={self.box_landing_phase}")

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

        if self.state == 'BOX_LANDING':
            parts.append(f"edges={len(self.edge_points)}")

        self.get_logger().info("Status: " + " | ".join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
