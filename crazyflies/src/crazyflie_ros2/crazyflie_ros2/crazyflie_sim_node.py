"""
ROS2 Crazyflie Simulation Node (updated to mirror crazyflie_node.py TF chain)

- Static TFs:  map -> world (identity), world -> odom (identity)
- Dynamic TF:  odom -> base (robot pose)
- Topic frames:
    /crazyflie/pose    : frame_id = <world_frame>
    /crazyflie/odom    : header.frame_id = <odom_frame>, child_frame_id = <base_frame>
    /crazyflie/scan    : frame_id = <base_frame>
    /crazyflie/range/* : frame_id = <base_frame>/{front,back,left,right,up,down}

Also keeps back-compat for users who only set 'frame_id': if 'base_frame' is default and
'frame_id' is changed, we keep them in sync (same behavior as crazyflie_node.py).
"""

import math
import threading
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range, LaserScan
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


@dataclass
class BoxDims:
    half_xy: float = 3.75  # box extends from -3.75..+3.75 in x and y (7.5m square)
    height: float = 2.0    # ceiling z = 2.0 m


@dataclass
class ObstaclePlatform:
    """Raised platform obstacle in the environment."""
    x_min: float = 0.85    # Platform at positive x, positive y
    x_max: float = 1.15    # 0.3m wide
    y_min: float = 0.85
    y_max: float = 1.15    # 0.3m deep
    height: float = 0.3    # 0.3m tall platform


@dataclass
class ObstacleCylinder:
    """Cylindrical obstacle in the environment."""
    center_x: float = -1.875  # Halfway between center (0,0) and corner (-3.75, -3.75)
    center_y: float = -1.875
    radius: float = 0.25      # 0.25m radius
    height: float = 0.3       # 0.3m tall cylinder


class CrazyflieSimNode(Node):
    def __init__(self):
        super().__init__('crazyflie_sim_node')

        # ----- Parameters -----
        self.declare_parameter('frame_id', 'crazyflie')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('takeoff_height', 0.30)
        self.declare_parameter('default_yaw_rate', 0.3)  # rad/s slow spin when idle
        self.declare_parameter('speed_limit', 1.0)       # m/s clamp
        self.declare_parameter('yaw_rate_limit', 2.0)    # rad/s clamp
        self.declare_parameter('box_half_xy', 3.75)      # increased by 50%
        self.declare_parameter('box_height', 2.0)
        self.declare_parameter('responsiveness', 0.3)    # velocity response lag (0-1, lower = more lag)

        # Explicit TF frames (mirror real node)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('odom_frame', 'crazyflie/odom')
        self.declare_parameter('base_frame', 'crazyflie')

        # Read parameters
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.takeoff_height = float(self.get_parameter('takeoff_height').value)
        self.default_yaw_rate = float(self.get_parameter('default_yaw_rate').value)
        self.speed_limit = float(self.get_parameter('speed_limit').value)
        self.yaw_rate_limit = float(self.get_parameter('yaw_rate_limit').value)
        self.responsiveness = float(self.get_parameter('responsiveness').value)
        self.box = BoxDims(
            half_xy=float(self.get_parameter('box_half_xy').value),
            height=float(self.get_parameter('box_height').value),
        )
        self.obstacle = ObstaclePlatform()  # Fixed 0.3x0.3m platform
        self.cylinder = ObstacleCylinder()  # Fixed cylinder with radius 0.25m

        self.map_frame   = self.get_parameter('map_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.odom_frame  = self.get_parameter('odom_frame').value
        self.base_frame  = self.get_parameter('base_frame').value

        # Back-compat: if user changed frame_id but not base_frame, keep them in sync
        if self.base_frame == 'crazyflie' and self.frame_id != 'crazyflie':
            self.base_frame = self.frame_id

        # QoS (match sensor streams style from real node)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ----- Publishers -----
        self.pose_pub = self.create_publisher(PoseStamped, '/crazyflie/pose', 10)
        self.odom_pub = self.create_publisher(Odometry, '/crazyflie/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/crazyflie/scan', 10)

        self.range_pubs = {
            'front': self.create_publisher(Range, '/crazyflie/range/front', sensor_qos),
            'back':  self.create_publisher(Range, '/crazyflie/range/back',  sensor_qos),
            'left':  self.create_publisher(Range, '/crazyflie/range/left',  sensor_qos),
            'right': self.create_publisher(Range, '/crazyflie/range/right', sensor_qos),
            'up':    self.create_publisher(Range, '/crazyflie/range/up',    sensor_qos),
            'down':  self.create_publisher(Range, '/crazyflie/range/down',  sensor_qos),
        }

        # ----- TF broadcasters -----
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_identities()

        # ----- Subscriber -----
        self.cmd_lock = threading.Lock()
        self.cmd_vel = Twist()
        self.last_cmd_time = self.get_clock().now()
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        # ----- Sim state -----
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0   # starts on floor
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._max_range = 3.49  # to mirror mapper expectation

        # Current velocities (for lag/inertia simulation)
        self.vx_current = 0.0
        self.vy_current = 0.0
        self.vz_current = 0.0
        self.yaw_rate_current = 0.0

        # Takeoff target
        self._target_z = self.takeoff_height

        # Timing
        self.dt = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(self.dt, self._update_and_publish)

        self.get_logger().info(
            f"CrazyflieSimNode started: box=({self.box.half_xy*2:.1f}m x {self.box.half_xy*2:.1f}m x {self.box.height:.1f}m), "
            f"takeoff={self.takeoff_height:.2f} m, rate={self.publish_rate_hz:.1f} Hz, responsiveness={self.responsiveness:.2f}"
        )
        self.get_logger().info(
            f"Obstacle platform at ({self.obstacle.x_min:.2f}-{self.obstacle.x_max:.2f}, "
            f"{self.obstacle.y_min:.2f}-{self.obstacle.y_max:.2f}), height={self.obstacle.height:.2f}m"
        )
        self.get_logger().info(
            f"Obstacle cylinder at ({self.cylinder.center_x:.2f}, {self.cylinder.center_y:.2f}), "
            f"radius={self.cylinder.radius:.2f}m, height={self.cylinder.height:.2f}m"
        )

    # -------------------- TF setup --------------------
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

    # -------------------- Callbacks --------------------
    def _cmd_vel_cb(self, msg: Twist):
        with self.cmd_lock:
            # Clamp for sanity
            msg.linear.x = max(-self.speed_limit, min(self.speed_limit, msg.linear.x))
            msg.linear.y = max(-self.speed_limit, min(self.speed_limit, msg.linear.y))
            msg.linear.z = max(-self.speed_limit, min(self.speed_limit, msg.linear.z))
            msg.angular.z = max(-self.yaw_rate_limit, min(self.yaw_rate_limit, msg.angular.z))
            self.cmd_vel = msg
            self.last_cmd_time = self.get_clock().now()

    # -------------------- Simulation core --------------------
    def _update_and_publish(self):
        now = self.get_clock().now()
        dt = self.dt

        # Read current command (body-frame velocities)
        with self.cmd_lock:
            cmd = self.cmd_vel

        # Default behavior if idle: hover at target_z and slowly yaw
        time_since_cmd = (now - self.last_cmd_time).nanoseconds * 1e-9
        idle = time_since_cmd > 0.5  # 0.5 s without command => idle

        vx_b_target = 0.0
        vy_b_target = 0.0
        vz_target = 0.0
        yaw_rate_target = self.default_yaw_rate if idle else cmd.angular.z

        if idle:
            # Passive altitude hold toward target_z
            vz_target = 1.0 * (self._target_z - self.z)  # simple P term
        else:
            vx_b_target = cmd.linear.x
            vy_b_target = cmd.linear.y
            vz_target = cmd.linear.z
            # Mimic hardware node's gentle target height adjustment semantics
            if abs(cmd.linear.z) < 1e-6:
                # no vertical command: hold target
                vz_target = 1.0 * (self._target_z - self.z)
            else:
                self._target_z = float(min(self.box.height - 0.05, max(0.05, self._target_z + cmd.linear.z * 0.01)))

        # Apply velocity lag/inertia (exponential smoothing to simulate real drone response)
        alpha = self.responsiveness  # 0.3 means slower response
        self.vx_current = alpha * vx_b_target + (1 - alpha) * self.vx_current
        self.vy_current = alpha * vy_b_target + (1 - alpha) * self.vy_current
        self.vz_current = alpha * vz_target + (1 - alpha) * self.vz_current
        self.yaw_rate_current = alpha * yaw_rate_target + (1 - alpha) * self.yaw_rate_current

        # Body -> world transform for velocities (using current yaw)
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        vx_w = c * self.vx_current - s * self.vy_current
        vy_w = s * self.vx_current + c * self.vy_current

        # Integrate
        self.x += vx_w * dt
        self.y += vy_w * dt
        self.z += self.vz_current * dt
        self.yaw += self.yaw_rate_current * dt

        # Constrain inside the box
        self._constrain_inside_box()

        # Publish data
        stamp = now.to_msg()
        self._publish_pose_odom_tf(stamp)
        self._publish_ranges_and_scan(stamp)

    def _constrain_inside_box(self):
        hx = self.box.half_xy
        hy = self.box.half_xy
        # Clip position to inside (tiny margin)
        self.x = min(hx - 0.001, max(-hx + 0.001, self.x))
        self.y = min(hy - 0.001, max(-hy + 0.001, self.y))
        self.z = min(self.box.height - 0.001, max(0.001, self.z))

    # -------------------- Publishing helpers --------------------
    def _publish_pose_odom_tf(self, stamp):
        # PoseStamped in world frame (for visualization; consistent with real node)
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.world_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.x, self.y, self.z
        qx, qy, qz, qw = self._quat_from_rpy(0.0, 0.0, self.yaw)
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = qx, qy, qz, qw
        self.pose_pub.publish(pose)

        # TF: odom -> base (dynamic)
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = stamp
        t_odom_base.header.frame_id = self.odom_frame
        t_odom_base.child_frame_id = self.base_frame
        t_odom_base.transform.translation.x, t_odom_base.transform.translation.y, t_odom_base.transform.translation.z = self.x, self.y, self.z
        t_odom_base.transform.rotation.x, t_odom_base.transform.rotation.y, t_odom_base.transform.rotation.z, t_odom_base.transform.rotation.w = qx, qy, qz, qw
        self.tf_broadcaster.sendTransform(t_odom_base)

        # Odometry must reflect odom->base
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose = pose.pose
        self.odom_pub.publish(odom)

    def _publish_ranges_and_scan(self, stamp):
        # Distances along body axes to walls (2D ray to axis-aligned square)
        front = self._ray_to_box_distance_2d(dir_bx=1.0, dir_by=0.0)   # +X body
        back  = self._ray_to_box_distance_2d(dir_bx=-1.0, dir_by=0.0)  # -X body
        left  = self._ray_to_box_distance_2d(dir_bx=0.0, dir_by=1.0)   # +Y body
        right = self._ray_to_box_distance_2d(dir_bx=0.0, dir_by=-1.0)  # -Y body
        up    = max(0.0, self.box.height - self.z)
        down  = self._compute_range_down()

        # Publish Range messages (IR-like) — use base_frame/* for sensor frames
        for name, dist in [
            ('front', front), ('back', back), ('left', left), ('right', right),
            ('up', up), ('down', down),
        ]:
            r = Range()
            r.header = Header(stamp=stamp, frame_id=f'{self.base_frame}/{name}')
            r.radiation_type = Range.INFRARED
            r.field_of_view = 0.436  # ~25 deg
            r.min_range = 0.01
            r.max_range = 4.0
            r.range = float('inf') if dist > r.max_range else float(max(r.min_range, dist))
            self.range_pubs[name].publish(r)

        # LaserScan: order [back, right, front, left]; frame_id = base_frame
        scan = LaserScan()
        scan.header = Header(stamp=stamp, frame_id=self.base_frame)
        scan.range_min = 0.01
        scan.range_max = self._max_range

        def clip_inf(d):
            return float('inf') if d > self._max_range else float(max(scan.range_min, d))

        scan.ranges = [clip_inf(back), clip_inf(right), clip_inf(front), clip_inf(left)]
        scan.angle_min = -0.5 * 2.0 * math.pi
        scan.angle_max = 0.25 * 2.0 * math.pi
        scan.angle_increment = 0.5 * math.pi  # 90 deg
        self.scan_pub.publish(scan)

    # -------------------- Geometry helpers --------------------
    def _compute_range_down(self) -> float:
        """
        Compute downward range distance accounting for obstacles (platform and cylinder).
        When drone is over an obstacle, measures to top of that obstacle.
        When crossing edge, creates spike in reading (floor drop).
        """
        # Check if drone is over the rectangular platform
        if (self.obstacle.x_min <= self.x <= self.obstacle.x_max and
            self.obstacle.y_min <= self.y <= self.obstacle.y_max):
            # Over the platform - measure to top of platform
            distance = max(0.0, self.z - self.obstacle.height)
        # Check if drone is over the cylindrical obstacle
        elif math.hypot(self.x - self.cylinder.center_x, self.y - self.cylinder.center_y) <= self.cylinder.radius:
            # Over the cylinder - measure to top of cylinder
            distance = max(0.0, self.z - self.cylinder.height)
        else:
            # Not over any obstacle - measure to floor
            distance = max(0.0, self.z - 0.0)

        return distance

    def _ray_to_cylinder_distance_2d(self, dx: float, dy: float) -> float:
        """
        Calculate distance from current position to cylinder along direction (dx, dy).
        Returns inf if no intersection.
        """
        # Ray-circle intersection in 2D
        # Ray: P(t) = (self.x, self.y) + t*(dx, dy)
        # Circle: (x - cx)^2 + (y - cy)^2 = r^2

        # Vector from circle center to ray origin
        fx = self.x - self.cylinder.center_x
        fy = self.y - self.cylinder.center_y

        # Quadratic coefficients: a*t^2 + b*t + c = 0
        a = dx*dx + dy*dy
        if a < 1e-9:  # Ray has no direction
            return float('inf')

        b = 2.0 * (fx*dx + fy*dy)
        c = fx*fx + fy*fy - self.cylinder.radius*self.cylinder.radius

        discriminant = b*b - 4*a*c

        if discriminant < 0:  # No intersection
            return float('inf')

        # Two intersection points
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)

        # We want the smallest positive t (closest intersection in ray direction)
        if t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return float('inf')  # Both intersections behind us

    def _ray_to_box_distance_2d(self, dir_bx: float, dir_by: float) -> float:
        """
        Distance from current (x,y,yaw) along BODY axis to nearest obstacle in world.
        Checks both box walls and cylindrical obstacle, returns minimum distance.
        """
        # Rotate body direction into world
        c = math.cos(self.yaw); s = math.sin(self.yaw)
        dx = c * dir_bx - s * dir_by
        dy = s * dir_bx + c * dir_by

        # Check box walls
        t_candidates = []
        hx = self.box.half_xy

        if abs(dx) > 1e-9:
            # Intersect with x = +hx and x = -hx
            t = ( hx - self.x) / dx
            if t > 0: t_candidates.append(t)
            t = (-hx - self.x) / dx
            if t > 0: t_candidates.append(t)

        if abs(dy) > 1e-9:
            # Intersect with y = +hx and y = -hx (same half extent)
            t = ( hx - self.y) / dy
            if t > 0: t_candidates.append(t)
            t = (-hx - self.y) / dy
            if t > 0: t_candidates.append(t)

        wall_dist = min(t_candidates) if t_candidates else float('inf')

        # Check cylinder obstacle
        cylinder_dist = self._ray_to_cylinder_distance_2d(dx, dy)

        # Return the closer of the two
        return min(wall_dist, cylinder_dist)

    @staticmethod
    def _quat_from_rpy(roll, pitch, yaw):
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * cp * cy
        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
