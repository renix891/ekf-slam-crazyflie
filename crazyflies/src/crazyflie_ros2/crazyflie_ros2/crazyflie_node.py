

import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import Range, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

offset = 0.15

class CrazyflieNode(Node):
    """ROS2 node for Crazyflie control and telemetry (with full TF chain)."""

    def __init__(self):
        super().__init__('crazyflie_node')

        self.motors_killed = False
        self._last_stop_send = 0
        self._last_blocked_warning = 0.0    

        # ---------- Parameters ----------
        self.declare_parameter('uri', 'radio://0/80/2M/E7E7E7E7E2')
        self.declare_parameter('hover_height', 0.6)
        self.declare_parameter('speed_factor', 0.6)

        # Back-compat: original code exposed 'frame_id' as the base frame
        self.declare_parameter('frame_id', 'crazyflie')
        self.declare_parameter('logging_only', False)

        # New: explicit TF frames to mirror the sim node
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('odom_frame', 'crazyflie/odom')
        self.declare_parameter('base_frame', 'crazyflie')

        # ---------- Get parameters ----------
        self.uri = self.get_parameter('uri').get_parameter_value().string_value
        self.hover_height = self.get_parameter('hover_height').get_parameter_value().double_value
        self.speed_factor = self.get_parameter('speed_factor').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.logging_only = self.get_parameter('logging_only').get_parameter_value().bool_value

        self.map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        # If user changed 'frame_id' but not 'base_frame', keep them in sync for BC.
        if self.base_frame == 'crazyflie' and self.frame_id != 'crazyflie':
            self.base_frame = self.frame_id

        # ---------- QoS profiles ----------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------- Publishers ----------
        self.pose_pub = self.create_publisher(PoseStamped, '/crazyflie/pose', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/crazyflie/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/crazyflie/odom', 10)

        self.range_pubs = {
            'front': self.create_publisher(Range, '/crazyflie/range/front', sensor_qos),
            'back':  self.create_publisher(Range, '/crazyflie/range/back',  sensor_qos),
            'left':  self.create_publisher(Range, '/crazyflie/range/left',  sensor_qos),
            'right': self.create_publisher(Range, '/crazyflie/range/right', sensor_qos),
            'up':    self.create_publisher(Range, '/crazyflie/range/up',    sensor_qos),
            'down':  self.create_publisher(Range, '/crazyflie/range/down',  sensor_qos),
        }

        # ---------- Subscribers ----------
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.emergency_stop_sub = self.create_subscription(Bool, '/crazyflie/emergency_stop', self.emergency_stop_callback, 10)

        # ---------- TF ----------
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_identities()  # map->world and world->odom (both identity)

        # ---------- Control state ----------
        self.cmd_vel = Twist()
        self.cmd_lock = threading.Lock()

        # ---------- Crazyflie init ----------
        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        # Setup callbacks
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)

        # Control timer (10Hz)
        self.control_timer = self.create_timer(0.1, self.send_control_command)

        # Connect to Crazyflie
        mode = "LOGGING ONLY (no flight commands)" if self.logging_only else "FULL CONTROL"
        self.get_logger().info(f'Mode: {mode}')
        self.get_logger().info(f'Connecting to Crazyflie at {self.uri}...')
        self.cf.open_link(self.uri)

    # ---------- Static TFs ----------

    def _publish_static_identities(self):
        now = self.get_clock().now().to_msg()

        # map -> world (identity)
        t_m_w = TransformStamped()
        t_m_w.header.stamp = now
        t_m_w.header.frame_id = self.map_frame
        t_m_w.child_frame_id = self.world_frame
        t_m_w.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t_m_w)

        # world -> odom (identity)
        t_w_o = TransformStamped()
        t_w_o.header.stamp = now
        t_w_o.header.frame_id = self.world_frame
        t_w_o.child_frame_id = self.odom_frame
        t_w_o.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t_w_o)

    # ---------- Subscriptions / Commands ----------

    def cmd_vel_callback(self, msg: Twist):
        """Handle incoming velocity commands."""
        if self.motors_killed:
            # Reject commands when e-stop is active (with throttled logging)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self._last_blocked_warning > 2.0:  # Log every 2 seconds
                self.get_logger().warn('Rejecting cmd_vel - emergency stop is active')
                self._last_blocked_warning = current_time
            return
        with self.cmd_lock:
            self.cmd_vel = msg

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop command - sends stop setpoint to kill motors."""
        if not msg.data:
            # Ignore false/reset messages - e-stop is permanent until node restart
            return

        if self.motors_killed:
            # E-stop already active, ignore duplicate commands
            return

        # First e-stop trigger
        self.get_logger().warn('Emergency stop triggered - killing motors and halting all commands')
        self.motors_killed = True

        try:
            self.cf.commander.send_stop_setpoint()
            self.get_logger().info('Stop setpoint sent to Crazyflie')
        except Exception as e:
            self.get_logger().error(f'Failed to send stop setpoint: {e}')

    def send_control_command(self):
        """Send velocity commands to Crazyflie at 10Hz."""
        if self.logging_only:
            return

        # If e-stop is active, keep sending stop setpoints to maintain connection
        if self.motors_killed:
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception as e:
                # Log errors but don't spam (already logged in emergency_stop_callback)
                pass
            return

        with self.cmd_lock:
            vx = self.cmd_vel.linear.x * self.speed_factor
            vy = self.cmd_vel.linear.y * self.speed_factor
            yaw_rate = math.degrees(self.cmd_vel.angular.z) * self.speed_factor

            # Height adjustment via small increments (hover setpoint)
            if abs(self.cmd_vel.linear.z) > 1e-9:
                self.hover_height += self.cmd_vel.linear.z * 0.1
                self.hover_height = max(0.1, min(1.0, self.hover_height))

        # Send hover setpoint
        self.cf.commander.send_hover_setpoint(vx, vy, yaw_rate, self.hover_height)

    # ---------- Crazyflie link callbacks ----------

    def _connected(self, uri):
        self.get_logger().info(f'Connected to {uri}')
        self._setup_position_logging()
        self._setup_sensor_logging()

    def _disconnected(self, uri):
        self.get_logger().info(f'Disconnected from {uri}')

    def _connection_failed(self, uri, msg):
        self.get_logger().error(f'Connection to {uri} failed: {msg}')

    # ---------- Logging setup ----------

    def _setup_position_logging(self):
        """Setup position and orientation logging from on-board estimator."""
        log_conf = LogConfig(name='Position', period_in_ms=100)
        log_conf.add_variable('stateEstimate.x', 'float')
        log_conf.add_variable('stateEstimate.y', 'float')
        log_conf.add_variable('stateEstimate.z', 'float')
        log_conf.add_variable('stabilizer.roll', 'float')
        log_conf.add_variable('stabilizer.pitch', 'float')
        log_conf.add_variable('stabilizer.yaw', 'float')

        try:
            self.cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._position_callback)
            log_conf.start()
            self.get_logger().info('Position logging started')
        except KeyError as e:
            self.get_logger().error(f'Could not start position logging: {e}')
        except AttributeError:
            self.get_logger().error('Could not add position log config')

    def _setup_sensor_logging(self):
        """Setup range sensor logging."""
        log_conf = LogConfig(name='Range', period_in_ms=100)
        log_conf.add_variable('range.front', 'uint16_t')
        log_conf.add_variable('range.back', 'uint16_t')
        log_conf.add_variable('range.left', 'uint16_t')
        log_conf.add_variable('range.right', 'uint16_t')
        log_conf.add_variable('range.up', 'uint16_t')
        log_conf.add_variable('range.zrange', 'uint16_t')

        try:
            self.cf.log.add_config(log_conf)
            log_conf.data_received_cb.add_callback(self._sensor_callback)
            log_conf.start()
            self.get_logger().info('Range sensor logging started')
        except KeyError as e:
            self.get_logger().error(f'Could not start sensor logging: {e}')
        except AttributeError:
            self.get_logger().error('Could not add sensor log config')

    # ---------- Telemetry callbacks ----------

    def _position_callback(self, timestamp, data, logconf):
        """Publish position/orientation -> Pose, TF (odom->base), and Odometry."""
        stamp = self.get_clock().now().to_msg()

        # Position (meters) & orientation (radians converted from degrees)
        x = data['stateEstimate.x']
        y = data['stateEstimate.y']
        z = data['stateEstimate.z']

        roll = math.radians(data['stabilizer.roll'])
        pitch = math.radians(data['stabilizer.pitch'])
        yaw = math.radians(data['stabilizer.yaw'])

        qx, qy, qz, qw = self._euler_to_quaternion(roll, pitch, yaw)

        # PoseStamped (visualization; world frame by convention)
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=stamp, frame_id=self.world_frame)
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.pose_pub.publish(pose_msg)

        # Dynamic TF: odom -> base (actual robot pose)
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = stamp
        t_odom_base.header.frame_id = self.odom_frame
        t_odom_base.child_frame_id = self.base_frame
        t_odom_base.transform.translation.x = x
        t_odom_base.transform.translation.y = y
        t_odom_base.transform.translation.z = z
        t_odom_base.transform.rotation.x = qx
        t_odom_base.transform.rotation.y = qy
        t_odom_base.transform.rotation.z = qz
        t_odom_base.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t_odom_base)

        # Odometry must reflect odom->base
        odom_msg = Odometry()
        odom_msg.header = Header(stamp=stamp, frame_id=self.odom_frame)
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose = pose_msg.pose
        self.odom_pub.publish(odom_msg)

    def _sensor_callback(self, timestamp, data, logconf):
        """Publish range sensors and a 4-beam LaserScan in base frame."""
        stamp = self.get_clock().now().to_msg()

        sensors = {
            'front': data['range.front'] + offset,
            'back':  data['range.back'] + offset,
            'left':  data['range.left'] + offset,
            'right': data['range.right'] + offset,
            'up':    data['range.up'],
            'down':  data['range.zrange']
        }

        # Range messages
        for sensor_name, distance_mm in sensors.items():
            r = Range()
            r.header = Header(stamp=stamp, frame_id=f'{self.base_frame}/{sensor_name}')
            r.radiation_type = Range.INFRARED
            r.field_of_view = 0.436  # ~25 deg
            r.min_range = 0.01
            r.max_range = 4.0
            r.range = (distance_mm / 1000.0) if distance_mm is not None else float('inf')
            self.range_pubs[sensor_name].publish(r)

        # LaserScan compatible with simple mapper (order: [back, right, front, left])
        max_range = 3.49  # meters

        def convert(distance_mm):
            d_m = (distance_mm / 1000.0) if distance_mm is not None else float('inf')
            return float('inf') if d_m > max_range else max(0.01, d_m)

        scan = LaserScan()
        scan.header = Header(stamp=stamp, frame_id=self.base_frame)
        scan.range_min = 0.01
        scan.range_max = max_range
        scan.ranges = [
            convert(data['range.back']),
            convert(data['range.right']),
            convert(data['range.front']),
            convert(data['range.left']),
        ]
        scan.angle_min = -0.5 * 2 * math.pi
        scan.angle_max =  0.25 * 2 * math.pi
        scan.angle_increment = 0.5 * math.pi
        self.scan_pub.publish(scan)

    # ---------- Helpers ----------

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * cp * cy
        return qx, qy, qz, qw

    def destroy_node(self):
        """Cleanup when node is destroyed."""
        self.get_logger().info('Shutting down Crazyflie node...')
        try:
            if self.cf is not None:
                self.cf.close_link()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()