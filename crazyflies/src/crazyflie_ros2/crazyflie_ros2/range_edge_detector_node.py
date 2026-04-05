# -*- coding: utf-8 -*-
import math, csv
from collections import deque
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import PointStamped

class RangeEdgeDetector(Node):
    def __init__(self):
        super().__init__('range_edge_detector')

        # ---- Params (match your best performer) ----
        self.declare_parameter('topic', '/crazyflie/range/down')
        self.declare_parameter('window', 20)           # rows
        self.declare_parameter('z_thresh', 3.8)
        self.declare_parameter('log_csv', True)
        self.declare_parameter('log_dir', '.')
        self.declare_parameter('expected_height_m', 0.30)  # informational

        self.topic = self.get_parameter('topic').value
        self.window = int(self.get_parameter('window').value)
        self.z_thresh = float(self.get_parameter('z_thresh').value)
        self.expected_h = float(self.get_parameter('expected_height_m').value)

        # ---- QoS ----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50
        )

        # ---- Subs/Pubs ----
        self.sub = self.create_subscription(Range, self.topic, self.cb, sensor_qos)
        self.edge_pub = self.create_publisher(Bool, '/perception/edge_event', 10)
        self.score_pub = self.create_publisher(Float32, '/perception/edge_score', 10)
        self.delta_pub = self.create_publisher(Float32, '/perception/edge_delta', 10)
        self.edge_point_pub = self.create_publisher(PointStamped, '/perception/edge_point', 10)

        # ---- Buffers ----
        self.samples = deque(maxlen=self.window)     # (t, r)
        self.deltas  = deque(maxlen=self.window)     # (t, dr/dt)
        self.last_edge = False
        self.msg_count = 0
        self.edge_count = 0

        # ---- Logging ----
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv = None
        if self.get_parameter('log_csv').value:
            path = f"{self.get_parameter('log_dir').value}/range_down_online_{ts}.csv"
            self.csv = open(path, 'w', newline='')
            self.writer = csv.writer(self.csv)
            self.writer.writerow(['time_ns','time_iso','range_m','height_err_m','delta_m_per_s','zscore_delta','edge_flag'])
            self.get_logger().info(f'Logging edges to {path}')

        self.get_logger().info(
            f'Edge detector initialized: listening on {self.topic}, '
            f'window={self.window}, z_thresh={self.z_thresh}, expected_height={self.expected_h}m')

    def now(self):
        return self.get_clock().now()

    def cb(self, msg: Range):
        t = self.now()
        tns = t.nanoseconds
        r = float(msg.range)
        if math.isnan(r) or math.isinf(r):
            return

        self.msg_count += 1

        # Log periodic status to show it's running
        if self.msg_count == 1:
            self.get_logger().info('Edge detector started - receiving range data')
        elif self.msg_count % 500 == 0:  # Every ~10 seconds at 50Hz
            self.get_logger().info(
                f'Edge detector active: {self.msg_count} range samples processed, '
                f'{self.edge_count} edges detected')

        # Append, compute derivative
        if self.samples:
            t0, r0 = self.samples[-1]
            dt = max(1e-6, (tns - t0) / 1e9)
            drdt = (r - r0) / dt
        else:
            drdt = 0.0

        self.samples.append((tns, r))
        self.deltas.append((tns, drdt))

        # Rolling mean/std of derivative
        vals = [v for (_, v) in self.deltas]
        n = len(vals)
        if n >= max(3, self.window // 3):
            mean_d = sum(vals)/n
            var_d = sum((v-mean_d)**2 for v in vals) / max(1, n-1)
            std_d = math.sqrt(var_d)
            z = 0.0 if std_d < 1e-9 else (drdt - mean_d)/std_d
        else:
            z = 0.0

        # Edge decision (rising-edge publish)
        edge_now = abs(z) >= self.z_thresh
        edge_msg = Bool(); edge_msg.data = (edge_now and not self.last_edge)
        self.last_edge = edge_now

        # Publish diagnostics always
        self.score_pub.publish(Float32(data=float(z)))
        self.delta_pub.publish(Float32(data=float(drdt)))

        if edge_msg.data:
            self.edge_count += 1
            self.edge_pub.publish(edge_msg)
            # Optionally publish a PointStamped "edge point" using current time in /map or /world if you fuse pose here.
            pt = PointStamped()
            pt.header.stamp = t.to_msg()
            pt.header.frame_id = 'world'  # adjust if you choose another frame
            # Leave zero; your nav node can stamp position using current pose when it receives edge_event.
            self.edge_point_pub.publish(pt)
            self.get_logger().warn(
                f'EDGE DETECTED #{self.edge_count} | z-score={z:.2f}, dr/dt={drdt:.3f}, range={r:.3f}m')

        # CSV log compatible w/ offline analyzer
        if self.csv:
            from datetime import datetime as _dt
            iso = _dt.utcnow().isoformat(timespec='milliseconds') + 'Z'
            h_err = r - self.expected_h
            self.writer.writerow([tns, iso, f'{r:.6f}', f'{h_err:.6f}', f'{drdt:.6f}', f'{z:.6f}', int(edge_now)])

    def destroy_node(self):
        try:
            if self.csv:
                self.csv.flush(); self.csv.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RangeEdgeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
