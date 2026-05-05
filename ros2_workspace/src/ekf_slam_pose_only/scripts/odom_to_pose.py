#!/usr/bin/env python3
"""Republish nav_msgs/Odometry on /crazyflie/odom as geometry_msgs/PoseStamped on /ekf_pose.

Used by the odom-only baseline launch so the planner/nav/box_landing nodes
(which subscribe to /ekf_pose) consume raw odometry instead of EKF-corrected pose.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class OdomToPose(Node):
    def __init__(self):
        super().__init__('odom_to_pose')
        self.pub = self.create_publisher(PoseStamped, '/ekf_pose', 10)
        self.sub = self.create_subscription(
            Odometry, '/crazyflie/odom', self.cb, 10)

    def cb(self, msg: Odometry):
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = 'map'
        ps.pose = msg.pose.pose
        self.pub.publish(ps)


def main():
    rclpy.init()
    rclpy.spin(OdomToPose())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
