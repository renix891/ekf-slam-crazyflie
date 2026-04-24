#!/usr/bin/env python3
"""Scripted takeoff → forward → yaw → sideways → hover → land for the Gazebo Crazyflie."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


# (label, linear.x, linear.y, linear.z, angular.z, duration_s)
FLIGHT_PLAN = [
    ('takeoff',  0.0, 0.0,  0.3, 0.0, 3.0),
    ('forward',  0.1, 0.0,  0.0, 0.0, 5.0),
    ('yaw',      0.0, 0.0,  0.0, 0.3, 3.0),
    ('sideways', 0.0, 0.1,  0.0, 0.0, 5.0),
    ('hover',    0.0, 0.0,  0.0, 0.0, 2.0),
    ('land',     0.0, 0.0, -0.2, 0.0, 3.0),
]
CMD_RATE_HZ = 20.0
PRE_TAKEOFF_DELAY = 2.0


class GzTakeoffDriver(Node):
    def __init__(self):
        super().__init__('gz_takeoff')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.enable_pub = self.create_publisher(Bool, '/crazyflie/enable', 10)

        self.current_cmd = Twist()
        self.create_timer(1.0 / CMD_RATE_HZ, self._publish_cmd)

        self._start_sequence()

    def _start_sequence(self):
        self.get_logger().info('arming motors')
        msg = Bool()
        msg.data = True
        # Publish a few times to make sure the bridge catches it.
        for _ in range(5):
            self.enable_pub.publish(msg)

        self.get_logger().info(f'waiting {PRE_TAKEOFF_DELAY}s before takeoff')
        self._phase_index = 0
        self._phase_deadline = None
        self.create_timer(PRE_TAKEOFF_DELAY, self._run_next_phase, callback_group=None)

    def _run_next_phase(self):
        # One-shot kickoff; subsequent phases are scheduled from the completion timer.
        if self._phase_index == 0 and self._phase_deadline is None:
            self._start_phase()

    def _start_phase(self):
        if self._phase_index >= len(FLIGHT_PLAN):
            self.get_logger().info('flight plan complete — shutting down')
            rclpy.shutdown()
            return

        label, vx, vy, vz, wz, duration = FLIGHT_PLAN[self._phase_index]
        self.get_logger().info(
            f'phase {self._phase_index+1}/{len(FLIGHT_PLAN)}: {label} '
            f'(vx={vx} vy={vy} vz={vz} wz={wz}) for {duration}s')

        self.current_cmd = Twist()
        self.current_cmd.linear.x = vx
        self.current_cmd.linear.y = vy
        self.current_cmd.linear.z = vz
        self.current_cmd.angular.z = wz

        self._phase_deadline = self.get_clock().now().nanoseconds * 1e-9 + duration
        self._phase_timer = self.create_timer(duration, self._finish_phase)

    def _finish_phase(self):
        self._phase_timer.cancel()
        self._phase_index += 1
        self._start_phase()

    def _publish_cmd(self):
        self.cmd_pub.publish(self.current_cmd)


def main():
    rclpy.init()
    node = GzTakeoffDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Best-effort zero command on exit.
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
