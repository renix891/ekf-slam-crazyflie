#!/usr/bin/env python3
"""Republish nav_msgs/Odometry on /crazyflie/odom as geometry_msgs/PoseStamped on /ekf_pose.

Used by the odom-only baseline launch so the planner/nav/box_landing nodes
(which subscribe to /ekf_pose) consume raw odometry instead of EKF-corrected pose.

For the final validation experiment this script also injects cumulative
Gaussian (Brownian) drift on (x, y, yaw) so the odom-only baseline behaves
like a real IMU-only pose estimate. The drift is integrated as a random
walk: each callback the per-axis offset is incremented by N(0, sigma * sqrt(dt)).
This makes the same noise level produce drift that grows with sqrt(t), as
real IMU bias does.

Disable noise (e.g. for sanity tests) by setting parameter
~enable_noise:=false or running with --ros-args -p enable_noise:=false.
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np


class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')

        # Per-axis drift sigma in (units / sqrt(s)). Defaults match the final
        # experiment spec: sigma_xy = 0.003 m per second cumulative,
        # sigma_yaw = 0.001 rad per second cumulative.
        self.sigma_xy_per_s   = self.declare_parameter(
            'sigma_xy_per_s', 0.003).value
        self.sigma_yaw_per_s  = self.declare_parameter(
            'sigma_yaw_per_s', 0.001).value
        self.enable_noise     = self.declare_parameter(
            'enable_noise', True).value
        self.seed             = self.declare_parameter('seed', -1).value
        if self.seed >= 0:
            self.rng = np.random.default_rng(int(self.seed))
        else:
            self.rng = np.random.default_rng()

        # Cumulative drift state (Brownian random walk).
        self.dx_drift = 0.0
        self.dy_drift = 0.0
        self.dyaw_drift = 0.0
        self.last_t = None

        self.pub = self.create_publisher(PoseStamped, '/ekf_pose', 10)
        self.sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self.cb, 10)

        self.get_logger().info(
            "odom_to_pose ready (noise={}, sigma_xy={:.4f} m/√s, "
            "sigma_yaw={:.4f} rad/√s)".format(
                self.enable_noise, self.sigma_xy_per_s, self.sigma_yaw_per_s))

    @staticmethod
    def _yaw_from_quat(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _quat_from_yaw(yaw):
        return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    def cb(self, msg: Odometry):
        # Compute dt from message stamps so the random walk scales correctly
        # with the actual /crazyflie/odom rate (Gazebo publishes ~50 Hz).
        h = msg.header.stamp
        t = h.sec + h.nanosec * 1e-9
        if self.last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self.last_t)
        self.last_t = t

        if self.enable_noise and dt > 0.0:
            sqrt_dt = math.sqrt(dt)
            self.dx_drift   += self.rng.normal(0.0, self.sigma_xy_per_s  * sqrt_dt)
            self.dy_drift   += self.rng.normal(0.0, self.sigma_xy_per_s  * sqrt_dt)
            self.dyaw_drift += self.rng.normal(0.0, self.sigma_yaw_per_s * sqrt_dt)

        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = 'map'

        # Position with cumulative drift.
        ps.pose.position.x = msg.pose.pose.position.x + self.dx_drift
        ps.pose.position.y = msg.pose.pose.position.y + self.dy_drift
        ps.pose.position.z = msg.pose.pose.position.z

        # Yaw with cumulative drift; preserve roll and pitch from the source.
        # (Roll/pitch are tiny on a hovering CF; our drift model only targets yaw.)
        if self.enable_noise:
            yaw = self._yaw_from_quat(msg.pose.pose.orientation) + self.dyaw_drift
            qx, qy, qz, qw = self._quat_from_yaw(yaw)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
        else:
            ps.pose.orientation = msg.pose.pose.orientation

        self.pub.publish(ps)


def main():
    rclpy.init()
    try:
        rclpy.spin(OdomToPose())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
