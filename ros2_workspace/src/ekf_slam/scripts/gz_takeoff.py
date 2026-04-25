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

# Planar flight plan: (label, vx, vy, wz, duration_s).  No linear.z — altitude
# is handled by the P-controller based on HOVER_HEIGHT.
#
# Square trajectory at 0.15 m/s in body x then y (yaw stays at 0 throughout, so
# body and world axes coincide and the path traces a 1.0 m square in world).
# 1.0 m / 0.15 m/s ≈ 6.67 s of pure translation per leg → plenty of distinct
# motion for the SVD scan matcher.  Hover holds between legs let the EKF
# settle and let the rotation/motion guards reset cleanly.
FLIGHT_PLAN = [
    # Body frame is FLU (REP-103): +x forward, +y left.  Yaw stays at 0 the
    # whole flight, so body == world axes — this traces a clockwise 1.0 m
    # square in the world frame: forward, right, back, left, returning to start.
    ('forward',    0.15,  0.0,  0.0, 6.67),
    ('hover_1',    0.0,   0.0,  0.0, 2.0),
    ('right',      0.0,  -0.15, 0.0, 6.67),
    ('hover_2',    0.0,   0.0,  0.0, 2.0),
    ('backward',  -0.15,  0.0,  0.0, 6.67),
    ('hover_3',    0.0,   0.0,  0.0, 2.0),
    ('left',       0.0,   0.15, 0.0, 6.67),
    ('hover_4',    0.0,   0.0,  0.0, 2.0),
]
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
        self.phase = 'arming'        # arming → climb → plan → land → done
        self.plan_idx = 0
        self.phase_t0 = None         # sim-time seconds when current phase started
        self.plan_cmd = (0.0, 0.0, 0.0)  # (vx, vy, wz) for current plan step
        self.target_alt = HOVER_HEIGHT  # altitude setpoint (ramped down during land)

        self.create_timer(1.0 / CMD_RATE_HZ, self._control_tick)

        # Arm motors — publish a few times so the ROS→GZ bridge catches at least one.
        arm = Bool()
        arm.data = True
        for _ in range(5):
            self.enable_pub.publish(arm)

        self.get_logger().info(
            f'armed; climbing to {HOVER_HEIGHT} m before planar flight plan')

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
                self.phase = 'plan'
                self.plan_idx = 0
                self.phase_t0 = now
                label, vx, vy, wz, _ = FLIGHT_PLAN[0]
                self.plan_cmd = (vx, vy, wz)
                self.get_logger().info(
                    f'reached {self.current_z:.2f} m — starting plan '
                    f'phase 1/{len(FLIGHT_PLAN)}: {label}')
            else:
                self._publish(0.0, 0.0, CLIMB_VZ, 0.0)
                return

        if self.phase == 'plan':
            label, vx, vy, wz, duration = FLIGHT_PLAN[self.plan_idx]
            elapsed = now - self.phase_t0
            if elapsed >= duration:
                self.plan_idx += 1
                if self.plan_idx >= len(FLIGHT_PLAN):
                    self.phase = 'land'
                    self.phase_t0 = now
                    self.get_logger().info('plan complete — starting land')
                else:
                    label, vx, vy, wz, _ = FLIGHT_PLAN[self.plan_idx]
                    self.plan_cmd = (vx, vy, wz)
                    self.phase_t0 = now
                    self.get_logger().info(
                        f'phase {self.plan_idx+1}/{len(FLIGHT_PLAN)}: {label} '
                        f'(vx={vx} vy={vy} wz={wz}) for {duration}s')
            else:
                vx, vy, wz = self.plan_cmd
                self._publish(vx, vy, self._altitude_hold_vz(), wz)
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
