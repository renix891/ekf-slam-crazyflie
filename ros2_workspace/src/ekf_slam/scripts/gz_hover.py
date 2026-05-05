#!/usr/bin/env python3
"""Minimal arm-and-hover script for the Gazebo Crazyflie.

Sequence:
  1. Publish True on /crazyflie/enable to arm motors.
  2. Climb at linear.z = CLIMB_VZ until altitude >= HOVER_HEIGHT.
  3. **Keep publishing zero-Twist** at CMD_RATE_HZ to actively hold the
     drone in place until HOLD_DEADLINE_S, then exit.

Why hold instead of releasing immediately on reaching altitude: with no
publisher on /cmd_vel, the simulated drone drifts horizontally a few cm
during the gap between hover-completion and nav-arm. That drift flips
the planner's tie-break (going north vs south of the obstacle) and
can route the drone the wrong way around the obstacle. Holding zero-
Twist pins the drone at (0, 0) so the planner sees the same start cell
every run.

HOLD_DEADLINE_S is set so the script exits a beat AFTER nav-arm so
navigation_node fully takes over without contention.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


HOVER_HEIGHT = 0.3      # m — must match navigation_node's flight_height default
CLIMB_VZ = 0.5          # m/s
CMD_RATE_HZ = 20.0
# Launch starts hover at t=3s and arms nav at t=12s. Hold zero-Twist until
# t=12.5s (= 9.5s of script lifetime) so the drone is pinned through the
# whole pre-goal interval. Nav takes over a half-second after we stop.
HOLD_DEADLINE_S = 9.5
# Hard timeout (kept for safety in case odom never arrives).
TIMEOUT_S = 12.0


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

        self._reached_hold_height = False

        self.create_timer(1.0 / CMD_RATE_HZ, self._tick)
        self.get_logger().info(
            f'armed; climbing to {HOVER_HEIGHT} m then holding zero-Twist '
            f'until t+{HOLD_DEADLINE_S:.1f}s (nav arms at t+9s)')

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

        # Hard exit after HOLD_DEADLINE_S — by this point nav has armed
        # (launch fires it at t+9s of hover lifetime) and we step out.
        if elapsed >= HOLD_DEADLINE_S:
            self._publish(0.0)
            self.get_logger().info(
                f'hold deadline reached at t+{elapsed:.2f}s — releasing '
                f'/cmd_vel to navigation_node')
            rclpy.shutdown()
            return

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

        if not self._reached_hold_height and self.current_z < HOVER_HEIGHT:
            self._publish(CLIMB_VZ)
            return

        if not self._reached_hold_height:
            self._reached_hold_height = True
            self.get_logger().info(
                f'reached {self.current_z:.2f} m — holding zero-Twist '
                f'until t+{HOLD_DEADLINE_S:.1f}s')

        # Active hold: keep publishing zero-Twist every tick. This actively
        # pins the drone in place during the pre-goal interval instead of
        # letting it drift uncontrolled.
        self._publish(0.0)


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
