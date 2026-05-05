#!/usr/bin/env python3
"""Hover-setpoint-style takeoff + planar flight plan for the Gazebo Crazyflie.

Mirrors the contract enforced by the real hardware stack:
- `autonomous_navigation_node.py` only commands (linear.x, linear.y, angular.z)
  for motion, using linear.z solely as a P-correction on altitude error.
- `crazyflie_node.py` forwards those to `cf.commander.send_hover_setpoint`,
  which handles altitude as a scalar setpoint, not a controlled axis.

The EKF-SLAM and planners assume the drone stays level at a fixed altitude
and the 4-beam lidar samples a horizontal slice. This script preserves that
assumption: climb vertically once, then only planar velocities.
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


HOVER_HEIGHT = 0.3         # target altitude for planar flight
CLIMB_VZ = 0.5             # m/s during initial takeoff
VZ_KP = 2.0                # altitude-hold P-gain (matches autonomous_nav default)
VZ_CLAMP = 0.2             # |linear.z| saturation during hover/flight (m/s)
CMD_RATE_HZ = 20.0

# Figure-8: two tangent circles traversed in opposite directions.
# Yaw stays at 0 throughout, so body == world axes — the (vx, vy) below are
# both body-frame commands and world-frame velocities.
#
#   speed v = 0.15 m/s, radius R = 0.5 m  ->  omega = v/R = 0.3 rad/s.
#   One full lap = 2*pi/omega ≈ 20.94 s; we run each lap for 20 s, which is
#   just shy of a full revolution and gives the EKF a continuous curving
#   trajectory (no discrete corners as in the prior square plan).
CIRCLE_SPEED = 0.15        # m/s tangential
CIRCLE_RADIUS = 0.5        # m
CIRCLE_OMEGA = CIRCLE_SPEED / CIRCLE_RADIUS  # rad/s
CIRCLE_DURATION = 20.0     # s per lap

LAND_DURATION = 5.0        # ramp target altitude from HOVER_HEIGHT → 0.0 over this


class GzTakeoffDriver(Node):
    def __init__(self):
        super().__init__('gz_takeoff')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/crazyflie/enable', 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self._on_odom, 10)

        # State
        self.current_z = 0.0
        self.have_odom = False
        self.phase = 'arming'        # arming → climb → circle_cw → circle_ccw → land → done
        self.phase_t0 = None         # sim-time seconds when current phase started
        self.target_alt = HOVER_HEIGHT  # altitude setpoint (ramped down during land)

        self.create_timer(1.0 / CMD_RATE_HZ, self._control_tick)

        # Arm motors — publish a few times so the ROS→GZ bridge catches at least one.
        arm = Bool()
        arm.data = True
        for _ in range(5):
            self.enable_pub.publish(arm)

        self.get_logger().info(
            f'armed; climbing to {HOVER_HEIGHT} m before figure-8 flight plan')

    def _on_odom(self, msg: Odometry):
        self.current_z = msg.pose.pose.position.z
        self.have_odom = True

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _altitude_hold_vz(self):
        """P-controller on altitude, saturated to ±VZ_CLAMP."""
        err = self.target_alt - self.current_z
        vz = VZ_KP * err
        if vz > VZ_CLAMP:
            vz = VZ_CLAMP
        elif vz < -VZ_CLAMP:
            vz = -VZ_CLAMP
        return vz

    def _publish(self, vx, vy, vz, wz):
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.linear.z = float(vz)
        cmd.angular.z = float(wz)
        self.cmd_pub.publish(cmd)

    def _circle_velocity(self, t, clockwise):
        """Body-frame (vx, vy) tracing a circle tangent to the +x axis at t=0.

        At t=0 the velocity is (CIRCLE_SPEED, 0).  The sign of vy decides the
        rotation direction:
            clockwise=True  -> drone curves toward -y (right)
            clockwise=False -> drone curves toward +y (left)
        Together these two laps form a figure-8 centered on the start point.
        """
        vx = CIRCLE_SPEED * math.cos(CIRCLE_OMEGA * t)
        vy = CIRCLE_SPEED * math.sin(CIRCLE_OMEGA * t)
        if clockwise:
            vy = -vy
        return vx, vy

    def _control_tick(self):
        if not self.have_odom:
            # Hold on the ground until we see odometry (so we know our altitude).
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        now = self._now_s()

        if self.phase == 'arming':
            # One-tick setup: transition straight to climb.
            self.phase = 'climb'
            self.phase_t0 = now
            self.get_logger().info('phase: climb')

        if self.phase == 'climb':
            # Drive straight vertical until we reach hover altitude.
            if self.current_z >= HOVER_HEIGHT:
                self.phase = 'circle_cw'
                self.phase_t0 = now
                self.get_logger().info(
                    f'reached {self.current_z:.2f} m — starting CW circle '
                    f'(R={CIRCLE_RADIUS} m, v={CIRCLE_SPEED} m/s)')
            else:
                self._publish(0.0, 0.0, CLIMB_VZ, 0.0)
                return

        if self.phase == 'circle_cw':
            elapsed = now - self.phase_t0
            if elapsed >= CIRCLE_DURATION:
                self.phase = 'circle_ccw'
                self.phase_t0 = now
                self.get_logger().info('CW lap done — starting CCW lap')
            else:
                vx, vy = self._circle_velocity(elapsed, clockwise=True)
                self._publish(vx, vy, self._altitude_hold_vz(), 0.0)
                return

        if self.phase == 'circle_ccw':
            elapsed = now - self.phase_t0
            if elapsed >= CIRCLE_DURATION:
                self.phase = 'land'
                self.phase_t0 = now
                self.get_logger().info('figure-8 complete — starting land')
            else:
                vx, vy = self._circle_velocity(elapsed, clockwise=False)
                self._publish(vx, vy, self._altitude_hold_vz(), 0.0)
                return

        if self.phase == 'land':
            elapsed = now - self.phase_t0
            frac = min(elapsed / LAND_DURATION, 1.0)
            # Ramp target altitude from HOVER_HEIGHT down to 0.0 over LAND_DURATION.
            self.target_alt = HOVER_HEIGHT * (1.0 - frac)
            self._publish(0.0, 0.0, self._altitude_hold_vz(), 0.0)
            if frac >= 1.0:
                self.phase = 'done'
                self.get_logger().info('landed — shutting down')
                rclpy.shutdown()
            return

        # Fallback for any unhandled phase state.
        self._publish(0.0, 0.0, 0.0, 0.0)


def main():
    rclpy.init()
    node = GzTakeoffDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
