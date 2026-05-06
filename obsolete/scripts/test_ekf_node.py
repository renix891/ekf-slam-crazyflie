#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped


LINEAR_SPEED = 0.1           # m/s on each side
ANGULAR_SPEED = 0.5          # rad/s at corners
TURN_DURATION = math.pi / 2 / ANGULAR_SPEED   # 0.5s per 90° turn? math: pi/2 / 0.5 = 3.14s
# Spec says "angular.z = 0.5 rad/s for 0.5 seconds to rotate 90 degrees".
# 0.5 rad/s * 0.5 s = 0.25 rad ≠ 90°, but we follow the spec literally.
TURN_DURATION = 0.5
SIDE_DURATION = 1.0 / LINEAR_SPEED   # 10s to traverse 1m at 0.1 m/s
LEG_PERIOD = SIDE_DURATION + TURN_DURATION

ODOM_RATE_HZ = 10.0
SCAN_RATE_HZ = 10.0
POSE_PRINT_PERIOD = 1.0
RUN_DURATION = 30.0
SCAN_RANGE = 0.5
SCAN_BEAMS = 4

# World-frame velocity for each of the 4 straight legs (+x, +y, -x, -y).
LEG_TWIST = [
    (+LINEAR_SPEED, 0.0),
    (0.0, +LINEAR_SPEED),
    (-LINEAR_SPEED, 0.0),
    (0.0, -LINEAR_SPEED),
]


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class EkfTestDriver(Node):
    def __init__(self):
        super().__init__('ekf_test_driver')

        self.odom_pub = self.create_publisher(Odometry, '/crazyflie/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/crazyflie/scan', 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/ekf_pose', self._on_pose, 10)

        self.start_time = self.get_clock().now()
        self.latest_pose = None

        # Integrated ground-truth pose published in odometry
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_update_time = 0.0

        self.create_timer(1.0 / ODOM_RATE_HZ, self._publish_odom)
        self.create_timer(1.0 / SCAN_RATE_HZ, self._publish_scan)
        self.create_timer(POSE_PRINT_PERIOD, self._print_pose)
        self.create_timer(0.1, self._check_shutdown)

        self.get_logger().info('EKF test driver started — running for 30s')

    def _elapsed(self):
        return (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

    def _current_twist(self, t):
        # Returns (vx_body, vy_body, omega) for the EKF's body-frame predict step.
        # Total cycle = 4 * LEG_PERIOD.
        cycle = 4.0 * LEG_PERIOD
        phase = t % cycle
        leg = int(phase // LEG_PERIOD)
        leg_t = phase - leg * LEG_PERIOD

        if leg_t < SIDE_DURATION:
            # Straight portion. Spec dictates body-frame components per leg.
            vx_world, vy_world = LEG_TWIST[leg]
            return vx_world, vy_world, 0.0
        else:
            # Turning portion at the corner.
            return 0.0, 0.0, ANGULAR_SPEED

    def _publish_odom(self):
        t = self._elapsed()
        vx, vy, omega = self._current_twist(t)

        # Integrate ground-truth pose from the twist so pose and twist stay consistent.
        if self.last_update_time > 0.0:
            dt = t - self.last_update_time
            self.x += vx * dt
            self.y += vy * dt
            self.yaw += omega * dt
        self.last_update_time = t

        qx, qy, qz, qw = yaw_to_quat(self.yaw)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = 0.0
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = omega

        self.odom_pub.publish(msg)

    def _publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi - (2.0 * math.pi / SCAN_BEAMS)
        msg.angle_increment = 2.0 * math.pi / SCAN_BEAMS
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / SCAN_RATE_HZ
        msg.range_min = 0.05
        msg.range_max = 5.0
        msg.ranges = [SCAN_RANGE] * SCAN_BEAMS
        msg.intensities = []
        self.scan_pub.publish(msg)

    def _on_pose(self, msg):
        self.latest_pose = msg

    def _print_pose(self):
        if self.latest_pose is None:
            self.get_logger().info('no /ekf_pose received yet')
            return
        p = self.latest_pose.pose.position
        q = self.latest_pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.get_logger().info(
            f'ekf_pose: x={p.x:+.3f} y={p.y:+.3f} yaw={math.degrees(yaw):+.1f}°')

    def _check_shutdown(self):
        if self._elapsed() >= RUN_DURATION:
            self.get_logger().info('30s elapsed — shutting down')
            rclpy.shutdown()


def main():
    rclpy.init()
    node = EkfTestDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
