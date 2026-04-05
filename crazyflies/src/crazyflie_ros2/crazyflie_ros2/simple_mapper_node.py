""" This simple mapper is loosely based on both the bitcraze cflib point cloud example
 https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/multiranger/multiranger_pointcloud.py
 and the webots epuck simple mapper example:
 https://github.com/cyberbotics/webots_ros2

 Originally from https://github.com/knmcguire/crazyflie_ros2_experimental/
 """

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

import math
import numpy as np
from bresenham import bresenham

class SimpleMapperMultiranger(Node):
    def __init__(self):
        super().__init__('simple_mapper_multiranger')

        # Declare parameters
        self.declare_parameter('robot_prefix', '/crazyflie')
        self.declare_parameter('avoidance_distance', 0.6)  # meters (increased from 0.5 to stay 0.1m further from obstacles)
        self.declare_parameter('max_avoidance_weight', 50)  # 1-50 range for weights
        self.declare_parameter('max_obstacle_range', 1.0)  # meters - ignore obstacles beyond this distance

        # Map size and origin parameters (Option B: extends in +X direction)
        self.declare_parameter('map_size_x', 40.0)  # meters (default 40m, was 20m)
        self.declare_parameter('map_size_y', 20.0)  # meters
        self.declare_parameter('map_origin_x', -10.0)  # meters (X range: -10 to +30)
        self.declare_parameter('map_origin_y', -10.0)  # meters (Y range: -10 to +10)
        self.declare_parameter('map_resolution', 0.1)  # meters per cell

        # Get parameter values
        robot_prefix = self.get_parameter('robot_prefix').value
        self.avoidance_distance = self.get_parameter('avoidance_distance').value
        self.max_avoidance_weight = self.get_parameter('max_avoidance_weight').value
        self.max_obstacle_range = self.get_parameter('max_obstacle_range').value

        # Map configuration
        self.map_size_x = self.get_parameter('map_size_x').value
        self.map_size_y = self.get_parameter('map_size_y').value
        self.map_origin_x = self.get_parameter('map_origin_x').value
        self.map_origin_y = self.get_parameter('map_origin_y').value
        self.map_resolution = self.get_parameter('map_resolution').value

        self.odom_subscriber = self.create_subscription(
            Odometry, robot_prefix + '/odom', self.odom_subscribe_callback, 10)
        self.ranges_subscriber = self.create_subscription(
            LaserScan, robot_prefix + '/scan', self.scan_subscribe_callback, 10)
        self.position = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.ranges = [0.0, 0.0, 0.0, 0.0]
        self.range_max = 3.5

        self.tfbr = StaticTransformBroadcaster(self)
        t_map = TransformStamped()
        t_map.header.stamp = self.get_clock().now().to_msg()
        t_map.header.frame_id = 'map'
        t_map.child_frame_id = robot_prefix + '/odom'
        t_map.transform.translation.x = 0.0
        t_map.transform.translation.y = 0.0
        t_map.transform.translation.z = 0.0
        self.tfbr.sendTransform(t_map)

        self.position_update = False

        # Calculate map dimensions
        self.map_width = int(self.map_size_x / self.map_resolution)
        self.map_height = int(self.map_size_y / self.map_resolution)
        map_size = self.map_width * self.map_height

        # Initialize occupancy grid (-1 = unknown)
        self.map = [-1] * map_size

        self.map_publisher = self.create_publisher(OccupancyGrid, robot_prefix + '/map',
                                                   qos_profile=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST,))

        self.get_logger().info(f"Simple mapper set for crazyflie " + robot_prefix +
                               f" using the odom and scan topic")
        self.get_logger().info(f"Map size: {self.map_size_x}x{self.map_size_y}m "
                               f"({self.map_width}x{self.map_height} cells)")
        self.get_logger().info(f"Map origin: ({self.map_origin_x}, {self.map_origin_y}), "
                               f"resolution: {self.map_resolution}m")
        self.get_logger().info(f"Avoidance distance: {self.avoidance_distance}m, "
                               f"max weight: {self.max_avoidance_weight}")
        self.get_logger().info(f"Max obstacle range: {self.max_obstacle_range}m "
                               f"(obstacles beyond this distance will be ignored)")
        self.publish_initial_map()

    def publish_initial_map(self):
        """Publish an initial map for testing."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.data = self.map
        self.map_publisher.publish(msg)
        self.get_logger().debug("Published initial empty map")

    def apply_ray_weights(self, ray_cells, obstacle_distance_cells, map_width, map_height):
        """
        Apply lightweight (integer) avoidance weights ONLY along the given ray,
        focusing near the obstacle end. Never overwrite non-empty cells (not 0 / -1).
        """
        if not ray_cells:
            return
        # Focus on last K cells near the obstacle (but on the robot side)
        # Use a small taper so closer-to-obstacle cells get larger weights.
        K = min(6, max(1, obstacle_distance_cells))  # up to ~6 cells
        start = max(0, len(ray_cells) - K - 1)
        window = ray_cells[start:-1] if len(ray_cells) > 1 else []

        # Avoid writing on the obstacle cell itself; we only weight the approach
        for i, (cx, cy) in enumerate(window):
            if 0 <= cx < map_width and 0 <= cy < map_height:
                idx = cy * map_width + cx
                current = self.map[idx]
                if current not in (-1, 0):
                    # Don't overwrite non-empty cells (occupied or previously weighted)
                    continue
                # Taper: nearest gets max, farther gets less (at least 1)
                # i=0 (farther) -> smaller; i=K-1 (near) -> larger
                w = max(1, int(round((i + 1) * self.max_avoidance_weight / K)))
                # Store as a small positive weight (1..max_avoidance_weight)
                self.map[idx] = min(self.max_avoidance_weight, w)

 
 # =======================================================================
 # Utility functions
 # =======================================================================
 
    def bresenham(x0: int, y0: int, x1: int, y1: int):
        """
        Integer Bresenham line from (x0,y0) to (x1,y1), inclusive of both endpoints.
        Returns a generator of (x,y) cells.
        """
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def euler_from_quaternion(self, x, y, z, w):
        """Convert quaternion to euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def odom_subscribe_callback(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.angles[0] = roll
        self.angles[1] = pitch
        self.angles[2] = yaw
        self.position_update = True

    def scan_subscribe_callback(self, msg):
        self.ranges = msg.ranges
        self.range_max = msg.range_max
        
        self.get_logger().debug(f"Received scan with {len(self.ranges)} ranges: {self.ranges}")
        
        if self.position_update is False:
            self.get_logger().warning("No position update yet, skipping scan")
            return
        
        self.get_logger().debug(f"Robot position: ({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})")
        
        data = self.rotate_and_create_points()
        self.get_logger().debug(f"Created {len(data)} obstacle points")

        points_x = []
        points_y = []

        # Use instance variables for map dimensions
        map_width = self.map_width
        map_height = self.map_height

        # Calculate current position in map coordinates
        position_x_map = int((self.position[0] - self.map_origin_x) / self.map_resolution)
        position_y_map = int((self.position[1] - self.map_origin_y) / self.map_resolution)
        
        self.get_logger().debug(f"Robot in map coords: ({position_x_map}, {position_y_map}), map size: {map_width}x{map_height}")
        
        # Check if robot position is within map bounds
        if not (0 <= position_x_map < map_width and 0 <= position_y_map < map_height):
            self.get_logger().error(f"Robot outside map bounds! Skipping scan.")
            return
        
        obstacle_count = 0

        for i in range(len(data)):
            point_x = int((data[i][0] - self.map_origin_x) / self.map_resolution)
            point_y = int((data[i][1] - self.map_origin_y) / self.map_resolution)

            # Check if point is within map bounds
            if not (0 <= point_x < map_width and 0 <= point_y < map_height):
                self.get_logger().debug(f"Point ({point_x}, {point_y}) outside bounds, skipping")
                continue

            points_x.append(point_x)
            points_y.append(point_y)
            obstacle_count += 1

            # Build the full ray once so we can both clear along it and apply weights on it
            ray_cells = list(bresenham(position_x_map, position_y_map, point_x, point_y))

            # Update cells along ray from robot to obstacle (free space)
            for line_x, line_y in ray_cells:
                if 0 <= line_x < map_width and 0 <= line_y < map_height:
                    map_idx = line_y * map_width + line_x
                    # Set to free ONLY if currently unknown or already free
                    # Do NOT overwrite any non-empty (weighted/occupied) cells
                    if self.map[map_idx] in (-1, 0):
                        self.map[map_idx] = 0

            # Mark detected obstacle point
            obstacle_idx = point_y * map_width + point_x
            self.map[obstacle_idx] = 100

            # Apply weighted avoidance ONLY along the same ray, near the obstacle
            obstacle_distance_cells = max(0, len(ray_cells) - 1)
            self.apply_ray_weights(ray_cells, obstacle_distance_cells, map_width, map_height)

        # (Removed old area/semicircle weighting; weights are now applied per-ray only)

        self.get_logger().debug(f"Processed {obstacle_count} obstacles, updating map")
        
        # Create and publish map message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self.map_resolution
        msg.info.width = map_width
        msg.info.height = map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.data = self.map
        
        self.get_logger().debug(f"Publishing map: {map_width}x{map_height}, {len(self.map)} cells")
        self.map_publisher.publish(msg)

    def rotate_and_create_points(self):
        data = []
        o = self.position
        roll = self.angles[0]
        pitch = self.angles[1]
        yaw = self.angles[2]
        r_back = self.ranges[0]
        r_right = self.ranges[1]
        r_front = self.ranges[2]
        r_left = self.ranges[3]
        
        self.get_logger().debug(f"Ranges - back: {r_back}, right: {r_right}, front: {r_front}, left: {r_left}, max: {self.range_max}")


        if (r_left < self.range_max and r_left != 0.0 and math.isinf(r_left) == False and r_left < self.max_obstacle_range):
            left = [o[0], o[1] + r_left, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, left))

        if (r_right < self.range_max and r_right != 0.0 and math.isinf(r_right) == False and r_right < self.max_obstacle_range):
            right = [o[0], o[1] - r_right, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, right))

        if (r_front < self.range_max and r_front != 0.0 and math.isinf(r_front) == False and r_front < self.max_obstacle_range):
            front = [o[0] + r_front, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, front))

        if (r_back < self.range_max and r_back != 0.0 and math.isinf(r_back) == False and r_back < self.max_obstacle_range):
            back = [o[0] - r_back, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, back))

        return data

    def rot(self, roll, pitch, yaw, origin, point):
        cosr = math.cos((roll))
        cosp = math.cos((pitch))
        cosy = math.cos((yaw))

        sinr = math.sin((roll))
        sinp = math.sin((pitch))
        siny = math.sin((yaw))

        roty = np.array([[cosy, -siny, 0],
                        [siny, cosy, 0],
                        [0, 0,    1]])

        rotp = np.array([[cosp, 0, sinp],
                        [0, 1, 0],
                        [-sinp, 0, cosp]])

        rotr = np.array([[1, 0,   0],
                        [0, cosr, -sinr],
                        [0, sinr,  cosr]])

        rotFirst = np.dot(rotr, rotp)

        rot = np.array(np.dot(rotFirst, roty))

        tmp = np.subtract(point, origin)
        tmp2 = np.dot(rot, tmp)
        return np.add(tmp2, origin)


def main(args=None):

    rclpy.init(args=args)
    simple_mapper_multiranger = SimpleMapperMultiranger()
    rclpy.spin(simple_mapper_multiranger)
    rclpy.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()