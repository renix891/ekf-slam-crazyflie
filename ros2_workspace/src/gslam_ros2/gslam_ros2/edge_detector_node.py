#!/usr/bin/env python3
"""
Edge Detector using Absolute Height Thresholds

Detects edges when the range sensor reading exceeds:
  flight_height + (box_height * modifier)

Simplified version without namespace prefixes for gslam_ros2 package.
"""
import math
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PointStamped


class EdgeDetectorNode(Node):
    def __init__(self):
        super().__init__('edge_detector_node')

        # ---- Parameters ----
        self.declare_parameter('topic', '/range/down')
        self.declare_parameter('flight_height', 0.5)       # Expected flight height (m)
        self.declare_parameter('box_height', 0.15)         # Expected box height (m)
        self.declare_parameter('modifier', 0.9)            # Threshold modifier (default 0.9)
        self.declare_parameter('edge_detection_delay', 5.0)  # Ignore edges for first N seconds
        self.declare_parameter('debounce_period', 0.3)     # Minimum time between edge detections (s)
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', '.')

        self.topic = self.get_parameter('topic').value
        self.flight_height = float(self.get_parameter('flight_height').value)
        self.box_height = float(self.get_parameter('box_height').value)
        self.modifier = float(self.get_parameter('modifier').value)
        self.edge_delay = float(self.get_parameter('edge_detection_delay').value)
        self.debounce = float(self.get_parameter('debounce_period').value)

        # Compute threshold
        # Edge detected when: range > upper_thresh
        self.upper_thresh = self.flight_height + (self.box_height * self.modifier)

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )

        # ---- Subscribers/Publishers ----
        self.sub = self.create_subscription(Range, self.topic, self.range_cb, sensor_qos)
        self.edge_pub = self.create_publisher(Bool, '/perception/edge_event', 10)
        self.score_pub = self.create_publisher(Float32, '/perception/edge_score', 10)
        self.delta_pub = self.create_publisher(Float32, '/perception/edge_delta', 10)
        self.edge_point_pub = self.create_publisher(PointStamped, '/perception/edge_point', 10)

        # ---- State ----
        self.start_time = None
        self.last_edge_time = None
        self.msg_count = 0
        self.edge_count = 0
        self.suppressed_count = 0
        self.last_edge_state = False

        # ---- Logging ----
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv = None
        if self.get_parameter('log_csv').value:
            path = f"{self.get_parameter('log_dir').value}/range_down_online_{ts}.csv"
            self.csv = open(path, 'w', newline='')
            self.writer = csv.writer(self.csv)
            self.writer.writerow([
                'time_ns', 'time_iso', 'range_m', 'height_err_m',
                'above_threshold', 'edge_flag'
            ])
            self.get_logger().info(f'Logging edges to {path}')

        self.get_logger().info(
            f'Edge Detector initialized: listening on {self.topic}')
        self.get_logger().info(
            f'  flight_height={self.flight_height:.3f}m, box_height={self.box_height:.3f}m, '
            f'modifier={self.modifier:.2f}')
        self.get_logger().info(
            f'  Edge detected when: range > {self.upper_thresh:.3f}m')
        self.get_logger().info(
            f'  edge_detection_delay={self.edge_delay:.1f}s, debounce={self.debounce:.2f}s')

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def range_cb(self, msg: Range):
        t_ns = self.get_clock().now().nanoseconds
        t_sec = t_ns / 1e9
        r = float(msg.range)

        # Initialize start time
        if self.start_time is None:
            self.start_time = t_sec

        if math.isnan(r) or math.isinf(r):
            return

        self.msg_count += 1

        # Log periodic status
        if self.msg_count == 1:
            self.get_logger().info('Edge detector started - receiving range data')
        elif self.msg_count % 500 == 0:  # Every ~10 seconds at 50Hz
            elapsed = t_sec - self.start_time
            status = f'Edge detector active: {self.msg_count} samples, {self.edge_count} edges'
            if elapsed < self.edge_delay:
                status += f', suppressed for {self.edge_delay - elapsed:.1f}s more'
            if self.suppressed_count > 0:
                status += f' ({self.suppressed_count} suppressed by delay)'
            self.get_logger().info(status)

        # Check if range is above threshold (edge condition)
        above_threshold = r > self.upper_thresh

        # Edge detection logic with rising edge detection and debounce
        edge_detected = False
        edge_suppressed = False

        # Check edge_detection_delay
        elapsed_time = t_sec - self.start_time
        if elapsed_time < self.edge_delay:
            if above_threshold and not self.last_edge_state:
                edge_suppressed = True
                self.suppressed_count += 1
        else:
            # Detect rising edge: transition from below to above threshold
            if above_threshold and not self.last_edge_state:
                # Check debounce period
                if self.last_edge_time is None or (t_sec - self.last_edge_time) >= self.debounce:
                    edge_detected = True

        # Update state for next iteration
        self.last_edge_state = above_threshold

        # Publish edge event
        if edge_detected:
            self.edge_count += 1
            self.last_edge_time = t_sec

            edge_msg = Bool()
            edge_msg.data = True
            self.edge_pub.publish(edge_msg)

            # Publish edge point
            pt = PointStamped()
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.header.frame_id = 'world'
            self.edge_point_pub.publish(pt)

            self.get_logger().warn(
                f'EDGE DETECTED #{self.edge_count} | range={r:.3f}m '
                f'(threshold: {self.upper_thresh:.3f}m)')

        # Publish diagnostics (always)
        # Use above_threshold as a binary score (1 if edge condition, 0 if normal)
        score = Float32()
        score.data = 1.0 if above_threshold else 0.0
        self.score_pub.publish(score)

        delta = Float32()
        delta.data = 0.0  # Not used anymore
        self.delta_pub.publish(delta)

        # CSV logging
        if self.csv:
            iso = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
            h_err = r - self.flight_height
            edge_flag = 1 if edge_detected else 0
            self.writer.writerow([
                t_ns, iso, f'{r:.6f}', f'{h_err:.6f}',
                int(above_threshold), edge_flag
            ])

    def destroy_node(self):
        try:
            if self.csv:
                self.csv.flush()
                self.csv.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EdgeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
