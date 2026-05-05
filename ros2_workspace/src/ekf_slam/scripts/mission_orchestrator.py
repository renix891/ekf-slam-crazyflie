#!/usr/bin/env python3
"""Two-leg mission orchestrator for the final validation experiment.

The OUTBOUND leg is armed by the launch file via two CLI ExecuteProcess
actions (`ros2 service call /enable_autonomous` and `ros2 topic pub --once
/goal_pose`) at t=12s — byte-for-byte identical to the proven
gazebo_full_nav.launch.py flight. This orchestrator does NOT touch the
first leg; it only watches for outbound landing and then drives the
return leg.

Sequence handled here:
  1. Wait for outbound landing (z < LAND_Z).
  2. Hover HOVER_S seconds.
  3. RETURN: enable_autonomous, publish /goal_pose at (0, 0, FLIGHT_Z).
  4. Wait for takeoff then landing (z < LAND_Z).
  5. Log final pose, idle until user Ctrl+C.

Continuous /ekf_pose is preserved across the brief outbound landing so
the EKF / odometry carry their accumulated state into the return leg —
the whole point of the experiment.
"""

from __future__ import annotations

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


GOAL_X    = 0.8
GOAL_Y    = 0.0
FLIGHT_Z  = 0.3
TAKEOFF_Z = 0.20      # m, "we're up" threshold
LAND_Z    = 0.05      # m, landing threshold (matches navigation_node)
HOVER_S   = 2.0       # seconds to wait after first touchdown
TICK_HZ   = 20.0      # bumped to 20Hz so the takeoff loop publishes /cmd_vel
                      # at the same rate gz_hover.py does
CLIMB_VZ  = 0.5       # m/s, matches gz_hover.py CLIMB_VZ


class MissionOrchestrator(Node):
    def __init__(self):
        super().__init__('mission_orchestrator')

        # NOTE: /goal_pose publisher is created lazily in publish_goal() right
        # before the return leg. Creating it at init caused the outbound
        # `ros2 topic pub --once /goal_pose` from the launch file to race
        # against this publisher's QoS discovery, and the planner sometimes
        # never received the one-shot outbound goal — the drone would just
        # hover. Lazy creation keeps the outbound publisher graph identical
        # to gazebo_full_nav.launch.py.
        self.goal_pub = None

        # /cmd_vel + /crazyflie/enable for the return-leg takeoff. After
        # outbound landing, navigation_node disables itself and the firmware
        # cuts motors. We need to re-arm and climb back to FLIGHT_Z before
        # nav can drive the return leg.
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/crazyflie/enable', 10)

        self.pose_sub = self.create_subscription(
            PoseStamped, '/ekf_pose', self.pose_cb, 10)

        self.nav_client = self.create_client(SetBool, '/enable_autonomous')

        # State machine — start at OUTBOUND. The launch file already
        # published the outbound goal and enabled nav; our job is to
        # watch for the outbound landing and drive the return leg.
        self.state = 'OUTBOUND'
        self.cur_z = 0.0
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.have_pose = False
        self.hover_started_at = None
        self.return_published = False
        self.landed_logged = False
        self.last_state = None
        # Wait until the drone has actually taken off before we treat a
        # low z reading as "landed" — orchestrator boots while drone is
        # still on the ground.
        self.has_been_airborne = False

        self.timer = self.create_timer(1.0 / TICK_HZ, self.tick)
        self.get_logger().info(
            "mission_orchestrator ready: outbound armed by launch (%.2f, %.2f); "
            "watching for landing → return (0, 0)" % (GOAL_X, GOAL_Y))

    def pose_cb(self, msg: PoseStamped):
        self.cur_z = msg.pose.position.z
        self.cur_x = msg.pose.position.x
        self.cur_y = msg.pose.position.y
        self.have_pose = True

    def publish_goal(self, x, y, z):
        # Lazy create — see __init__ for why.
        if self.goal_pub is None:
            self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'map'
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0
        self.goal_pub.publish(ps)
        self.get_logger().info(f"[orch] published /goal_pose ({x:.2f}, {y:.2f}, {z:.2f})")

    def call_set_bool(self, client, value, label):
        if not client.wait_for_service(timeout_sec=0.1):
            return False
        req = SetBool.Request()
        req.data = value
        client.call_async(req)
        self.get_logger().info(f"[orch] {label}={value}")
        return True

    def tick(self):
        if self.state != self.last_state:
            self.get_logger().info(f"[orch] state → {self.state}")
            self.last_state = self.state

        if not self.have_pose:
            return

        if self.state == 'OUTBOUND':
            # First wait until the drone is actually in the air, then watch
            # for landing. Without the airborne latch we'd flag z<LAND_Z
            # immediately on the ground.
            if not self.has_been_airborne:
                if self.cur_z >= TAKEOFF_Z:
                    self.has_been_airborne = True
                return
            if self.cur_z < LAND_Z:
                self.get_logger().info(
                    "[orch] OUTBOUND landed at (%.3f, %.3f). "
                    "Hovering %.1fs before return leg." %
                    (self.cur_x, self.cur_y, HOVER_S))
                self.hover_started_at = self.get_clock().now()
                self.state = 'HOVER'

        elif self.state == 'HOVER':
            elapsed = (self.get_clock().now() - self.hover_started_at).nanoseconds * 1e-9
            if elapsed >= HOVER_S:
                # Re-arm motors and start climbing. nav stays disabled so
                # this orchestrator owns /cmd_vel for the takeoff phase.
                arm = Bool()
                arm.data = True
                for _ in range(5):
                    self.enable_pub.publish(arm)
                self.get_logger().info(
                    "[orch] return-leg takeoff: re-armed motors, climbing to "
                    f"{FLIGHT_Z:.2f} m")
                self.state = 'RETURN_TAKEOFF'

        elif self.state == 'RETURN_TAKEOFF':
            # Publish climb commands every tick until the drone reaches
            # FLIGHT_Z. Once airborne, hand off to nav with the (0, 0) goal.
            if self.cur_z < FLIGHT_Z:
                cmd = Twist()
                cmd.linear.z = CLIMB_VZ
                self.cmd_pub.publish(cmd)
                return
            # Reached altitude. Publish a single zero-Twist to settle, then
            # arm nav and publish the return goal.
            self.cmd_pub.publish(Twist())
            if not self.call_set_bool(
                    self.nav_client, True, 'enable_autonomous'):
                return
            self.publish_goal(0.0, 0.0, FLIGHT_Z)
            self.return_published = True
            self.state = 'RETURN'

        elif self.state == 'RETURN':
            # Drone is already at flight altitude from the takeoff phase, so
            # advance immediately into RETURN_FLYING. Keep the >= TAKEOFF_Z
            # check as a safety in case the drone briefly dipped.
            if self.cur_z >= TAKEOFF_Z:
                self.state = 'RETURN_FLYING'

        elif self.state == 'RETURN_FLYING':
            if self.cur_z < LAND_Z:
                self.state = 'DONE'

        elif self.state == 'DONE':
            if not self.landed_logged:
                err = math.hypot(self.cur_x, self.cur_y)
                self.get_logger().info(
                    "[orch] RETURN landed at (%.3f, %.3f) — "
                    "Euclidean distance from (0, 0) = %.3f m (%.1f cm). "
                    "Mission complete." %
                    (self.cur_x, self.cur_y, err, err * 100.0))
                self.landed_logged = True
            # Stay idle until user Ctrl+C.


def main():
    rclpy.init()
    try:
        rclpy.spin(MissionOrchestrator())
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
