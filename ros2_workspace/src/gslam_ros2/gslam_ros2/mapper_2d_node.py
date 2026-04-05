#!/usr/bin/env python3
"""Simple 2D occupancy grid mapper using range sensors.

Based on the bitcraze cflib point cloud example and webots epuck mapper.
Originally from https://github.com/knmcguire/crazyflie_ros2_experimental/

Simplified for basic 2D occupancy mapping without avoidance features.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

import math
import numpy as np
from bresenham import bresenham


class Mapper2D(Node):
    def __init__(self):
        super().__init__('mapper_2d')

        # Declare parameters
        self.declare_parameter('map_size_x', 20.0)  # meters
        self.declare_parameter('map_size_y', 20.0)  # meters
        self.declare_parameter('map_origin_x', -10.0)  # meters
        self.declare_parameter('map_origin_y', -10.0)  # meters
        self.declare_parameter('map_resolution', 0.1)  # meters per cell

        # Map configuration
        self.map_size_x = self.get_parameter('map_size_x').value
        self.map_size_y = self.get_parameter('map_size_y').value
        self.map_origin_x = self.get_parameter('map_origin_x').value
        self.map_origin_y = self.get_parameter('map_origin_y').value
        self.map_resolution = self.get_parameter('map_resolution').value

        # Subscribe to odometry and laser scan
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # State variables
        self.position = [0.0, 0.0, 0.0]
        self.angles = [0.0, 0.0, 0.0]
        self.ranges = []
        self.range_max = 3.5
        self.position_update = False

        # Calculate map dimensions
        self.map_width = int(self.map_size_x / self.map_resolution)
        self.map_height = int(self.map_size_y / self.map_resolution)
        map_size = self.map_width * self.map_height

        # Initialize occupancy grid (-1 = unknown, 0 = free, 100 = occupied)
        self.map = [-1] * map_size

        # Publisher for occupancy grid
        self.map_publisher = self.create_publisher(
            OccupancyGrid,
            '/map',
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST
            )
        )

        # Broadcast static transform from world to odom frame
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        t_world_odom = TransformStamped()
        t_world_odom.header.stamp = self.get_clock().now().to_msg()
        t_world_odom.header.frame_id = 'world'
        t_world_odom.child_frame_id = 'odom'
        t_world_odom.transform.translation.x = 0.0
        t_world_odom.transform.translation.y = 0.0
        t_world_odom.transform.translation.z = 0.0
        t_world_odom.transform.rotation.x = 0.0
        t_world_odom.transform.rotation.y = 0.0
        t_world_odom.transform.rotation.z = 0.0
        t_world_odom.transform.rotation.w = 1.0  # Identity quaternion
        self.tf_broadcaster.sendTransform(t_world_odom)

        self.get_logger().info(f"2D Mapper initialized")
        self.get_logger().info(f"Map size: {self.map_size_x}x{self.map_size_y}m "
                               f"({self.map_width}x{self.map_height} cells)")
        self.get_logger().info(f"Map origin: ({self.map_origin_x}, {self.map_origin_y}), "
                               f"resolution: {self.map_resolution}m")
        self.get_logger().info(f"Broadcasting static TF: world -> odom")

        self.publish_initial_map()

    def publish_initial_map(self):
        """Publish an initial empty map."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.data = self.map
        self.map_publisher.publish(msg)
        self.get_logger().debug("Published initial empty map")

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

    def odom_callback(self, msg):
        """Update robot position from odometry."""
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.angles[0] = roll
        self.angles[1] = pitch
        self.angles[2] = yaw
        self.position_update = True

    def scan_callback(self, msg):
        """Process laser scan and update occupancy grid."""
        self.ranges = msg.ranges
        self.range_max = msg.range_max

        self.get_logger().debug(f"Received scan with {len(self.ranges)} ranges")

        if not self.position_update:
            self.get_logger().warning("No position update yet, skipping scan")
            return

        self.get_logger().debug(f"Robot position: ({self.position[0]:.2f}, "
                                f"{self.position[1]:.2f}, {self.position[2]:.2f})")

        # Create obstacle points from range data
        data = self.rotate_and_create_points()
        self.get_logger().debug(f"Created {len(data)} obstacle points")

        # Calculate current position in map coordinates
        position_x_map = int((self.position[0] - self.map_origin_x) / self.map_resolution)
        position_y_map = int((self.position[1] - self.map_origin_y) / self.map_resolution)

        self.get_logger().debug(f"Robot in map coords: ({position_x_map}, {position_y_map}), "
                                f"map size: {self.map_width}x{self.map_height}")

        # Check if robot position is within map bounds
        if not (0 <= position_x_map < self.map_width and
                0 <= position_y_map < self.map_height):
            self.get_logger().error(f"Robot outside map bounds! Skipping scan.")
            return

        obstacle_count = 0

        for i in range(len(data)):
            point_x = int((data[i][0] - self.map_origin_x) / self.map_resolution)
            point_y = int((data[i][1] - self.map_origin_y) / self.map_resolution)

            # Check if point is within map bounds
            if not (0 <= point_x < self.map_width and 0 <= point_y < self.map_height):
                self.get_logger().debug(f"Point ({point_x}, {point_y}) outside bounds, skipping")
                continue

            obstacle_count += 1

            # Get cells along ray from robot to obstacle using Bresenham
            ray_cells = list(bresenham(position_x_map, position_y_map, point_x, point_y))

            # Mark cells along ray as free space (except the last one)
            for line_x, line_y in ray_cells[:-1]:
                if 0 <= line_x < self.map_width and 0 <= line_y < self.map_height:
                    map_idx = line_y * self.map_width + line_x
                    # Only update if currently unknown
                    if self.map[map_idx] == -1:
                        self.map[map_idx] = 0

            # Mark detected obstacle point as occupied
            obstacle_idx = point_y * self.map_width + point_x
            self.map[obstacle_idx] = 100

        self.get_logger().debug(f"Processed {obstacle_count} obstacles, updating map")

        # Create and publish map message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.data = self.map

        self.get_logger().debug(f"Publishing map: {self.map_width}x{self.map_height}, "
                                f"{len(self.map)} cells")
        self.map_publisher.publish(msg)

    def rotate_and_create_points(self):
        """Create 3D points from range data with rotation applied."""
        data = []
        o = self.position
        roll = self.angles[0]
        pitch = self.angles[1]
        yaw = self.angles[2]

        # Assuming standard 4-sensor configuration: [back, right, front, left]
        if len(self.ranges) < 4:
            self.get_logger().warning(f"Expected 4 ranges, got {len(self.ranges)}")
            return data

        r_back = self.ranges[0]
        r_right = self.ranges[1]
        r_front = self.ranges[2]
        r_left = self.ranges[3]

        self.get_logger().debug(f"Ranges - back: {r_back:.2f}, right: {r_right:.2f}, "
                                f"front: {r_front:.2f}, left: {r_left:.2f}, max: {self.range_max}")

        # Process each valid range reading
        if r_left < self.range_max and r_left != 0.0 and not math.isinf(r_left):
            left = [o[0], o[1] + r_left, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, left))

        if r_right < self.range_max and r_right != 0.0 and not math.isinf(r_right):
            right = [o[0], o[1] - r_right, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, right))

        if r_front < self.range_max and r_front != 0.0 and not math.isinf(r_front):
            front = [o[0] + r_front, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, front))

        if r_back < self.range_max and r_back != 0.0 and not math.isinf(r_back):
            back = [o[0] - r_back, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, back))

        return data

    def rot(self, roll, pitch, yaw, origin, point):
        """Apply 3D rotation to a point."""
        cosr = math.cos(roll)
        cosp = math.cos(pitch)
        cosy = math.cos(yaw)

        sinr = math.sin(roll)
        sinp = math.sin(pitch)
        siny = math.sin(yaw)

        roty = np.array([[cosy, -siny, 0],
                        [siny, cosy, 0],
                        [0, 0, 1]])

        rotp = np.array([[cosp, 0, sinp],
                        [0, 1, 0],
                        [-sinp, 0, cosp]])

        rotr = np.array([[1, 0, 0],
                        [0, cosr, -sinr],
                        [0, sinr, cosr]])

        rotFirst = np.dot(rotr, rotp)
        rot = np.array(np.dot(rotFirst, roty))

        tmp = np.subtract(point, origin)
        tmp2 = np.dot(rot, tmp)
        return np.add(tmp2, origin)


def main(args=None):
    rclpy.init(args=args)
    mapper_2d = Mapper2D()
    rclpy.spin(mapper_2d)
    mapper_2d.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
