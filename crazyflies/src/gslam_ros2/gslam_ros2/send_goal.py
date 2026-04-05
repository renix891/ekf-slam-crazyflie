#!/usr/bin/env python3
"""
Simple script to send a goal pose to mission_manager_node.

Usage:
  ros2 run gslam_ros2 send_goal --x 1.0 --y 1.0
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import argparse


def main(args=None):
    parser = argparse.ArgumentParser(description='Send goal pose to mission manager')
    parser.add_argument('--x', type=float, default=1.0, help='Goal X position (meters)')
    parser.add_argument('--y', type=float, default=1.0, help='Goal Y position (meters)')
    parser.add_argument('--z', type=float, default=0.35, help='Goal Z position (meters)')

    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = Node('send_goal')

    goal_pub = node.create_publisher(PoseStamped, '/goal_pose', 10)

    # Wait for subscriber
    import time
    time.sleep(0.5)

    # Create goal message
    goal_msg = PoseStamped()
    goal_msg.header.stamp = node.get_clock().now().to_msg()
    goal_msg.header.frame_id = 'world'
    goal_msg.pose.position.x = parsed_args.x
    goal_msg.pose.position.y = parsed_args.y
    goal_msg.pose.position.z = parsed_args.z
    goal_msg.pose.orientation.w = 1.0

    goal_pub.publish(goal_msg)
    node.get_logger().info(f'Published goal: ({parsed_args.x:.2f}, {parsed_args.y:.2f}, {parsed_args.z:.2f})')

    time.sleep(0.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
