"""
D* Lite Path Planning Simple Node for Crazyflie (Triggered Replanning)

Similar to dstarlite_path_planning_node but doesn't continuously replan.
Instead, replanning is triggered via service call (e.g., after scanning phase).
Waypoints are simplified to reduce count.

Subscribes to:
  /crazyflie/map (nav_msgs/OccupancyGrid)
  /crazyflie/pose (geometry_msgs/PoseStamped)
  /goal_pose (geometry_msgs/PoseStamped)

Publishes:
  /planned_path (nav_msgs/Path) - simplified waypoints for autonomous navigation

Services:
  /trigger_replanning (std_srvs/Trigger) - trigger path replanning
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

import numpy as np
import math
import threading
from typing import Optional, Tuple, List


class DStarNode:
    """Node class for D* Lite algorithm."""
    def __init__(self, x: int = 0, y: int = 0, cost: float = 0.0):
        self.x = x
        self.y = y
        self.cost = cost


def add_coordinates(node1: DStarNode, node2: DStarNode) -> DStarNode:
    """Add coordinates of two nodes."""
    new_node = DStarNode()
    new_node.x = node1.x + node2.x
    new_node.y = node1.y + node2.y
    new_node.cost = node1.cost + node2.cost
    return new_node


def compare_coordinates(node1: DStarNode, node2: DStarNode) -> bool:
    """Check if two nodes have the same coordinates."""
    return node1.x == node2.x and node1.y == node2.y


class DStarLitePlanner:
    """D* Lite path planning algorithm for occupancy grids."""

    # 8-directional movement with diagonal costs
    motions = [
        DStarNode(1, 0, 1),
        DStarNode(0, 1, 1),
        DStarNode(-1, 0, 1),
        DStarNode(0, -1, 1),
        DStarNode(1, 1, 1),
        DStarNode(1, -1, 1),
        DStarNode(-1, 1, 1),
        DStarNode(-1, -1, 1)
    ]

    def __init__(self, width: int, height: int, logger):
        self.logger = logger
        self.x_max = width
        self.y_max = height

        # D* Lite state
        self.start = DStarNode(0, 0)
        self.goal = DStarNode(0, 0)
        self.U = []  # Priority queue
        self.km = 0.0
        self.kold = 0.0
        self.rhs = self.create_grid(math.inf)
        self.g = self.create_grid(math.inf)

        # Occupancy grid and obstacle tracking
        self.cost_map = None
        self.initialized = False

        # For path computation
        self.current_path = []
        self.planning_lock = threading.Lock()

    def create_grid(self, val: float) -> np.ndarray:
        """Create a grid initialized with a value."""
        return np.full((self.x_max, self.y_max), val, dtype=np.float64)

    def set_cost_map(self, cost_map: np.ndarray):
        """Update the cost map from occupancy grid."""
        with self.planning_lock:
            old_cost_map = self.cost_map
            self.cost_map = cost_map.copy()

            # If we have a previous cost map, detect changes
            if old_cost_map is not None and self.initialized:
                changed_cells = np.argwhere(old_cost_map != cost_map)
                if len(changed_cells) > 0:
                    self.logger.debug(f"Detected {len(changed_cells)} changed cells, will replan when triggered...")
                    self.handle_cost_changes(changed_cells)

    def handle_cost_changes(self, changed_cells: np.ndarray):
        """Handle changes in cost map by updating affected vertices."""
        if not self.initialized:
            return

        last = DStarNode(self.start.x, self.start.y)
        self.km += self.h(last)

        for cell in changed_cells:
            x, y = cell[0], cell[1]
            if x < 0 or x >= self.x_max or y < 0 or y >= self.y_max:
                continue
            u = DStarNode(x, y)
            # Update vertex and its neighbors
            self.update_vertex(u)
            for neighbor in self.get_neighbours(u):
                self.update_vertex(neighbor)

    def c(self, node1: DStarNode, node2: DStarNode) -> float:
        """Cost of moving from node1 to node2."""
        if self.cost_map is None:
            return math.inf

        # Check if node2 is within bounds
        if not self.is_valid(node2):
            return math.inf

        # Get cost from cost map
        cost = self.cost_map[node2.x, node2.y]

        # If cost is infinite (obstacle), can't traverse
        if np.isinf(cost):
            return math.inf

        # Calculate motion cost
        new_node = DStarNode(node1.x - node2.x, node1.y - node2.y)
        detected_motion = [m for m in self.motions if compare_coordinates(m, new_node)]

        if not detected_motion:
            return math.inf

        # Total cost = motion cost + cell cost
        return detected_motion[0].cost * cost

    def h(self, s: DStarNode) -> float:
        """Heuristic function (Chebyshev distance)."""
        return max(abs(self.start.x - s.x), abs(self.start.y - s.y))

    def calculate_key(self, s: DStarNode) -> Tuple[float, float]:
        """Calculate priority key for a node."""
        min_g_rhs = min(self.g[s.x, s.y], self.rhs[s.x, s.y])
        return (min_g_rhs + self.h(s) + self.km, min_g_rhs)

    def is_valid(self, node: DStarNode) -> bool:
        """Check if node is within grid bounds."""
        return 0 <= node.x < self.x_max and 0 <= node.y < self.y_max

    def get_neighbours(self, u: DStarNode) -> List[DStarNode]:
        """Get valid neighbors of a node."""
        return [add_coordinates(u, motion) for motion in self.motions
                if self.is_valid(add_coordinates(u, motion))]

    def pred(self, u: DStarNode) -> List[DStarNode]:
        """Get predecessors of a node."""
        return self.get_neighbours(u)

    def succ(self, u: DStarNode) -> List[DStarNode]:
        """Get successors of a node."""
        return self.get_neighbours(u)

    def initialize(self, start: DStarNode, goal: DStarNode):
        """Initialize D* Lite search."""
        goal_changed = self.initialized and not compare_coordinates(self.goal, goal)

        self.start = DStarNode(start.x, start.y)
        self.goal = DStarNode(goal.x, goal.y)

        if not self.initialized or goal_changed:
            if goal_changed:
                self.logger.info(f'Goal changed, reinitializing D* Lite')
            else:
                self.logger.info('D* Lite initialized')

            self.initialized = True
            self.U = []
            self.km = 0.0
            self.rhs = self.create_grid(math.inf)
            self.g = self.create_grid(math.inf)
            self.rhs[self.goal.x, self.goal.y] = 0
            self.U.append((self.goal, self.calculate_key(self.goal)))
            self.U.sort(key=lambda x: x[1])

    def update_vertex(self, u: DStarNode):
        """Update the rhs value and priority queue for a vertex."""
        if not compare_coordinates(u, self.goal):
            # Update rhs as minimum cost among successors
            min_cost = math.inf
            for sprime in self.succ(u):
                cost = self.c(u, sprime) + self.g[sprime.x, sprime.y]
                if cost < min_cost:
                    min_cost = cost
            self.rhs[u.x, u.y] = min_cost

        # Remove u from priority queue if present
        self.U = [(node, key) for node, key in self.U
                  if not compare_coordinates(node, u)]

        # If inconsistent, add to priority queue
        if self.g[u.x, u.y] != self.rhs[u.x, u.y]:
            self.U.append((u, self.calculate_key(u)))
            self.U.sort(key=lambda x: x[1])

    def compare_keys(self, key1: Tuple[float, float],
                     key2: Tuple[float, float]) -> bool:
        """Compare two priority keys."""
        return key1[0] < key2[0] or (key1[0] == key2[0] and key1[1] < key2[1])

    def compute_shortest_path(self):
        """Compute shortest path using D* Lite."""
        if not self.initialized or self.cost_map is None:
            return

        self.U.sort(key=lambda x: x[1])

        while (len(self.U) > 0 and
               (self.compare_keys(self.U[0][1], self.calculate_key(self.start)) or
                self.rhs[self.start.x, self.start.y] != self.g[self.start.x, self.start.y])):

            kold = self.U[0][1]
            u = self.U[0][0]
            self.U.pop(0)

            if self.compare_keys(kold, self.calculate_key(u)):
                self.U.append((u, self.calculate_key(u)))
                self.U.sort(key=lambda x: x[1])
            elif self.g[u.x, u.y] > self.rhs[u.x, u.y]:
                self.g[u.x, u.y] = self.rhs[u.x, u.y]
                for s in self.pred(u):
                    self.update_vertex(s)
            else:
                self.g[u.x, u.y] = math.inf
                for s in self.pred(u) + [u]:
                    self.update_vertex(s)

            self.U.sort(key=lambda x: x[1])

    def compute_path(self, start_grid: Tuple[int, int],
                     goal_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute path from start to goal."""
        with self.planning_lock:
            if self.cost_map is None:
                return []

            start_node = DStarNode(start_grid[0], start_grid[1])
            goal_node = DStarNode(goal_grid[0], goal_grid[1])

            # Initialize if needed or if goal changed
            if not self.initialized or not compare_coordinates(self.goal, goal_node):
                self.initialize(start_node, goal_node)

            # Update start position
            self.start = start_node

            # Compute shortest path
            self.compute_shortest_path()

            # Extract path
            if np.isinf(self.g[self.start.x, self.start.y]):
                self.logger.warning("No path found - start is unreachable")
                return []

            path = []
            current = DStarNode(self.start.x, self.start.y)
            max_steps = self.x_max * self.y_max  # Prevent infinite loops
            steps = 0

            while not compare_coordinates(current, self.goal) and steps < max_steps:
                path.append((current.x, current.y))

                # Find best successor
                successors = self.succ(current)
                if not successors:
                    self.logger.warning("No successors found")
                    break

                best_succ = min(successors,
                               key=lambda s: self.c(current, s) + self.g[s.x, s.y])

                # Check if we're making progress
                if np.isinf(self.c(current, best_succ) + self.g[best_succ.x, best_succ.y]):
                    self.logger.warning("Path blocked")
                    break

                current = best_succ
                steps += 1

            if compare_coordinates(current, self.goal):
                path.append((self.goal.x, self.goal.y))
                self.logger.debug(f"D* Lite path found with {len(path)} waypoints")
            else:
                self.logger.warning(f"Path computation stopped after {steps} steps")

            self.current_path = path
            return path


class DStarLitePathPlanningSimpleNode(Node):
    """ROS2 node for triggered D* Lite path planning on occupancy grids."""

    def __init__(self):
        super().__init__('dstarlite_path_planning_node')  # Keep same name for compatibility

        # Parameters
        self.declare_parameter('flight_height', 0.5)
        self.declare_parameter('waypoint_skip_distance', 0.3)  # Minimum distance between waypoints

        self.flight_height = self.get_parameter('flight_height').value
        self.waypoint_skip_distance = self.get_parameter('waypoint_skip_distance').value

        # State
        self.current_pose = None
        self.goal_pose = None
        self.occupancy_grid_msg = None
        self.planner = None

        # QoS
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/crazyflie/map', self.map_callback, map_qos)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/crazyflie/pose', self.pose_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_callback, 10)

        # Publishers
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)

        # Services
        self.replan_srv = self.create_service(
            Trigger, '/trigger_replanning', self.replan_callback)

        # Status timer (no continuous planning)
        self.status_timer = self.create_timer(5.0, self.status_loop)

        self.get_logger().info('D* Lite Path Planning Simple Node initialized (trigger-based)')
        self.get_logger().info('Waiting for occupancy grid on /crazyflie/map...')

    # ----- Callbacks -----
    def map_callback(self, msg: OccupancyGrid):
        """Handle occupancy grid updates."""
        was_none = self.occupancy_grid_msg is None
        self.occupancy_grid_msg = msg

        # Initialize planner if first map received
        if was_none:
            self.planner = DStarLitePlanner(
                msg.info.width,
                msg.info.height,
                self.get_logger()
            )
            self.get_logger().info(
                f'Received first occupancy grid: {msg.info.width}x{msg.info.height}, '
                f'res={msg.info.resolution}m')

        # Convert occupancy grid to cost map
        if self.planner is not None:
            cost_map = self._occupancy_to_cost_map(msg)
            self.planner.set_cost_map(cost_map)

    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates."""
        was_none = self.current_pose is None
        self.current_pose = msg
        if was_none:
            self.get_logger().debug(
                f'Received first pose: ({msg.pose.position.x:.2f}, '
                f'{msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})')

    def goal_callback(self, msg: PoseStamped):
        """Handle goal updates."""
        self.goal_pose = msg
        self.get_logger().info(
            f'New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})')
        self._publish_goal_marker()

        # Automatically plan initial path when goal is set
        self._do_replanning()

    def replan_callback(self, request, response):
        """Service callback to trigger replanning."""
        self.get_logger().info('Replanning triggered by service call')

        success = self._do_replanning()

        response.success = success
        if success:
            path_len = len(self.planner.current_path) if self.planner and self.planner.current_path else 0
            response.message = f"Replanning successful - {path_len} waypoints generated"
        else:
            response.message = "Replanning failed - check logs"

        return response

    # ----- Planning -----
    def _do_replanning(self) -> bool:
        """Execute path planning."""
        if not (self.planner and self.occupancy_grid_msg and
                self.current_pose and self.goal_pose):
            self.get_logger().warning("Cannot plan: missing planner, map, pose, or goal")
            return False

        start_world = (self.current_pose.pose.position.x,
                      self.current_pose.pose.position.y)
        goal_world = (self.goal_pose.pose.position.x,
                     self.goal_pose.pose.position.y)

        start_grid = self._world_to_grid(*start_world)
        goal_grid = self._world_to_grid(*goal_world)

        if start_grid is None:
            clamped = self._clamp_to_grid_bounds(*start_world)
            if clamped:
                start_grid = self._world_to_grid(*clamped)
            if start_grid is None:
                self.get_logger().warning("Start position out of bounds")
                return False

        if goal_grid is None:
            clamped = self._clamp_to_grid_bounds(*goal_world)
            if clamped:
                goal_grid = self._world_to_grid(*clamped)
            if goal_grid is None:
                self.get_logger().warning("Goal position out of bounds")
                return False

        # Compute path
        path = self.planner.compute_path(start_grid, goal_grid)

        # Simplify and publish path
        if path:
            simplified_path = self._simplify_path(path)
            self.get_logger().info(
                f"Path simplified: {len(path)} -> {len(simplified_path)} waypoints")
            self._publish_path(simplified_path)
            return True
        else:
            self.get_logger().warning("No path found")
            return False

    def _simplify_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Simplify path by skipping waypoints that are too close together.
        Always keeps first and last waypoint.
        """
        if len(path) <= 2:
            return path

        simplified = [path[0]]  # Always include start

        for i in range(1, len(path) - 1):
            gx_prev, gy_prev = simplified[-1]
            gx_curr, gy_curr = path[i]

            # Convert to world coordinates to check distance
            wx_prev, wy_prev = self._grid_to_world(gx_prev, gy_prev)
            wx_curr, wy_curr = self._grid_to_world(gx_curr, gy_curr)

            dist = math.hypot(wx_curr - wx_prev, wy_curr - wy_prev)

            # Only add waypoint if it's far enough from the last added one
            if dist >= self.waypoint_skip_distance:
                simplified.append(path[i])

        # Always include goal
        if not simplified or simplified[-1] != path[-1]:
            simplified.append(path[-1])

        return simplified

    def status_loop(self):
        """Periodic status logging."""
        parts = []
        if self.current_pose is None:
            parts.append("NO_POSE")
        else:
            parts.append(f"pose=({self.current_pose.pose.position.x:.1f},"
                        f"{self.current_pose.pose.position.y:.1f})")

        if self.occupancy_grid_msg is None:
            parts.append("NO_MAP")
        else:
            parts.append(f"map={self.occupancy_grid_msg.info.width}x"
                        f"{self.occupancy_grid_msg.info.height}")

        if self.goal_pose is None:
            parts.append("NO_GOAL")
        else:
            parts.append(f"goal=({self.goal_pose.pose.position.x:.1f},"
                        f"{self.goal_pose.pose.position.y:.1f})")

        if self.planner and self.planner.current_path:
            parts.append(f"path={len(self.planner.current_path)} waypoints")
        else:
            parts.append("NO_PATH")

        self.get_logger().debug("Status: " + " | ".join(parts))

    # ----- Utilities -----
    def _occupancy_to_cost_map(self, grid_msg: OccupancyGrid) -> np.ndarray:
        """Convert occupancy grid to cost map for D* Lite."""
        width = grid_msg.info.width
        height = grid_msg.info.height
        grid_array = np.array(grid_msg.data).reshape((height, width))

        # Initialize cost map
        cost_map = np.ones((width, height), dtype=np.float32)

        # Convert occupancy values to costs:
        # -1 (unknown): slight penalty (2.0)
        # 0 (free): low cost (1.0)
        # 1-50 (weighted): scale to higher costs
        # >50 (obstacle): infinite cost

        for y in range(height):
            for x in range(width):
                val = grid_array[y, x]

                if val < 0:  # Unknown
                    cost_map[x, y] = 2.0
                elif val == 0:  # Free
                    cost_map[x, y] = 1.0
                elif val <= 50:  # Weighted avoidance zone
                    # Scale 1-50 to 2.0-10.0
                    cost_map[x, y] = 2.0 + (val / 50.0) * 8.0
                else:  # Obstacle
                    cost_map[x, y] = np.inf

        return cost_map

    def _world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to grid coordinates."""
        if self.occupancy_grid_msg is None:
            return None
        info = self.occupancy_grid_msg.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            return (gx, gy)
        return None

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        info = self.occupancy_grid_msg.info
        wx = gx * info.resolution + info.origin.position.x
        wy = gy * info.resolution + info.origin.position.y
        return (wx, wy)

    def _clamp_to_grid_bounds(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Clamp world coordinates to grid bounds."""
        if self.occupancy_grid_msg is None:
            return None
        info = self.occupancy_grid_msg.info
        min_x = info.origin.position.x
        max_x = info.origin.position.x + (info.width * info.resolution)
        min_y = info.origin.position.y
        max_y = info.origin.position.y + (info.height * info.resolution)
        cx = max(min_x, min(x, max_x - info.resolution))
        cy = max(min_y, min(y, max_y - info.resolution))
        return (cx, cy)

    def _publish_path(self, path: List[Tuple[int, int]]):
        """Publish path as nav_msgs/Path."""
        if not path or self.occupancy_grid_msg is None:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = self.occupancy_grid_msg.header.frame_id

        for gx, gy in path:
            wx, wy = self._grid_to_world(gx, gy)
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.position.z = self.flight_height
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)

        self.path_pub.publish(path_msg)

    def _publish_goal_marker(self):
        """Publish goal marker for visualization in RViz."""
        if self.goal_pose is None:
            return

        marker = Marker()
        marker.header.frame_id = self.goal_pose.header.frame_id if self.goal_pose.header.frame_id else 'world'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position
        marker.pose.position.x = self.goal_pose.pose.position.x
        marker.pose.position.y = self.goal_pose.pose.position.y
        marker.pose.position.z = self.goal_pose.pose.position.z
        marker.pose.orientation.w = 1.0

        # Scale (size of sphere)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Color (green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        # Lifetime (0 = forever)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.goal_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = DStarLitePathPlanningSimpleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
