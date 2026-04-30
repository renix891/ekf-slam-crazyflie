#!/usr/bin/env python3
"""Minimal arm-and-hover script for the Gazebo Crazyflie.

Hands /cmd_vel control to the navigation stack as soon as hover altitude is
reached. Unlike gz_takeoff.py this does NO horizontal motion and exits
within ~5 s, so navigation_node can drive the drone without contention.

Sequence:
  1. Publish True on /crazyflie/enable to arm motors.
  2. Climb at linear.z = CLIMB_VZ until altitude >= HOVER_HEIGHT.
  3. Publish a single zero-Twist (release the throttle) and shut down.

Hard timeout: 8 s. If altitude is never reported, exit anyway so the launch
file can proceed.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


HOVER_HEIGHT = 0.3      # m — must match navigation_node's flight_height default
CLIMB_VZ = 0.5          # m/s
CMD_RATE_HZ = 20.0
TIMEOUT_S = 8.0         # absolute upper bound on script lifetime


class GzHover(Node):
    def __init__(self):
        super().__init__('gz_hover')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/crazyflie/enable', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self._on_odom, 10)

        self.current_z = 0.0
        self.have_odom = False
        self.t0 = self.get_clock().now().nanoseconds * 1e-9

        # Arm motors. Spam a few times so the ROS→GZ bridge catches at least one.
        arm = Bool()
        arm.data = True
        for _ in range(5):
            self.enable_pub.publish(arm)

        self.create_timer(1.0 / CMD_RATE_HZ, self._tick)
        self.get_logger().info(
            f'armed; climbing to {HOVER_HEIGHT} m then handing off /cmd_vel')

    def _on_odom(self, msg: Odometry):
        self.current_z = msg.pose.pose.position.z
        self.have_odom = True

    def _publish(self, vz):
        cmd = Twist()
        cmd.linear.z = float(vz)
        self.cmd_pub.publish(cmd)

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self.t0

        if elapsed > TIMEOUT_S:
            self.get_logger().warn(
                f'timeout after {elapsed:.1f}s (z={self.current_z:.2f}); '
                'releasing /cmd_vel anyway')
            self._publish(0.0)
            rclpy.shutdown()
            return

        if not self.have_odom:
            self._publish(0.0)
            return

        if self.current_z < HOVER_HEIGHT:
            self._publish(CLIMB_VZ)
            return

        # Reached hover altitude — release /cmd_vel and exit so navigation
        # node has uncontested control.
        self._publish(0.0)
        self.get_logger().info(
            f'reached {self.current_z:.2f} m — handing off to navigation_node')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = GzHover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
