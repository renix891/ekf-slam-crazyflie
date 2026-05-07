#!/usr/bin/env python3
"""TF broadcaster for bag-replay visualization in RViz.

The recorded bags do NOT contain /tf or /tf_static, so RViz cannot resolve
transforms between the frames the messages live in. This node fills the
gap by publishing the minimum tree RViz needs to render every recorded
topic top-down:

    map  (fixed)
     ├── crazyflie/odom                            (static, identity — truth
     │                                              and belief share start)
     └── base_link                                 (dynamic, from /ekf_pose)
          └── crazyflie/crazyflie/body/multiranger (static, identity)
               └── crazyflie/crazyflie/body/down_range  (static, identity)

The multiranger and down_range body offsets are below cm-scale and do not
matter for a top-down 2D recording. If a future use case needs them
exact, load the URDF instead.

Run from a launch file with use_sim_time:=True so transforms are stamped
on bag time, matching the consumer messages. The ExecuteProcess in
replay_for_video.launch.py does this.
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def make_static(parent: str, child: str, stamp) -> TransformStamped:
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    # Identity translation/rotation.
    t.transform.translation.x = 0.0
    t.transform.translation.y = 0.0
    t.transform.translation.z = 0.0
    t.transform.rotation.x = 0.0
    t.transform.rotation.y = 0.0
    t.transform.rotation.z = 0.0
    t.transform.rotation.w = 1.0
    return t


class BagTFBroadcaster(Node):
    def __init__(self):
        super().__init__('bag_tf_broadcaster')

        self._dyn = TransformBroadcaster(self)
        self._static = StaticTransformBroadcaster(self)

        # Static identity transforms are published once on startup; tf2
        # stores them with infinite validity so any-time TF lookup works.
        now = self.get_clock().now().to_msg()
        statics = [
            make_static('map',       'crazyflie/odom',                              now),
            make_static('base_link', 'crazyflie/crazyflie/body/multiranger',        now),
            make_static('crazyflie/crazyflie/body/multiranger',
                        'crazyflie/crazyflie/body/down_range',                      now),
        ]
        self._static.sendTransform(statics)
        self.get_logger().info(
            f"Published {len(statics)} static transforms (map, multiranger, down_range).")

        self._sub = self.create_subscription(
            PoseStamped, '/ekf_pose', self._on_pose, 10)
        self.get_logger().info(
            "Subscribed to /ekf_pose; will broadcast map -> base_link.")

        self._n = 0

    def _on_pose(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self._dyn.sendTransform(t)
        self._n += 1
        if self._n == 1:
            self.get_logger().info("First map -> base_link transform broadcast.")


def main():
    rclpy.init()
    node = BagTFBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
