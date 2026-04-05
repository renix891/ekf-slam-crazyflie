#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int8
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range, LaserScan
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

import logging
import time
import threading
import math
import os
import functools
from concurrent.futures import ThreadPoolExecutor

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# State machine constants
STATE_UNINITIALIZED = -1
STATE_FLYING = 0
STATE_LANDING = 1
STATE_LANDED = 2
STATE_TAKING_OFF = 3

# Control mode constants
CONTROL_MODE_VELOCITY = 0
CONTROL_MODE_POSITION = 1

class CrazyflieCommunicator(Node):
    def __init__(self, stockFirmware=False):
        super().__init__('crazyflie_communicator')

        # Declare and get parameters
        self.declare_parameter('height', 0.5)
        self.declare_parameter('address', 'radio://0/80/2M/E7E7E7E7E2')
        self.declare_parameter('linear_speed_factor', 0.5)
        self.declare_parameter('angular_speed_factor', 90.0)
        self.declare_parameter('control_rate_hz', 30.0)
        self.declare_parameter('position_tolerance', 0.05)
        # Frames and sensor model (to match odometry_node/ranger_node)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('min_range', 0.02)   # meters
        self.declare_parameter('max_range', 3.5)    # meters
        self.declare_parameter('field_of_view', 0.436)  # ~25 deg

        self.height = self.get_parameter('height').value
        address = self.get_parameter('address').value
        self.linear_speed_factor = self.get_parameter('linear_speed_factor').value
        self.angular_speed_factor = self.get_parameter('angular_speed_factor').value
        self.control_rate_hz = self.get_parameter('control_rate_hz').value
        self.position_tolerance = self.get_parameter('position_tolerance').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.fov = float(self.get_parameter('field_of_view').value)

        self.twist = Twist()

        # Command buffer for one-shot command execution
        self.cmd_buffer = Twist()
        self.cmd_lock = threading.Lock()

        # Position control state
        self.control_mode = CONTROL_MODE_VELOCITY
        self.position_target = None
        self.position_lock = threading.Lock()
        self.position_reached = False  # Track if we're within tolerance

        # Current odometry
        self.current_odom = None
        self.odom_lock = threading.Lock()

        # State machine
        self.state = STATE_UNINITIALIZED  # Start in uninitialized state
        self.state_lock = threading.Lock()

        # Create publishers FIRST (before connecting to Crazyflie)
        self.setpoint_pub = self.create_publisher(Twist, 'hover_setpoint', 1)
        self.position_reached_pub = self.create_publisher(Bool, 'position_target_reached', 1)
        self.state_pub = self.create_publisher(Int8, '/crazyflie/state', 1)

        # Odometry & range topics consistent with odometry_node.py and ranger_node.py
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.range_pubs = {
            'front': self.create_publisher(Range, 'range/front', 10),
            'back':  self.create_publisher(Range, 'range/back', 10),
            'left':  self.create_publisher(Range, 'range/left', 10),
            'right': self.create_publisher(Range, 'range/right', 10),
            'up':    self.create_publisher(Range, 'range/up', 10),
        }
        # TF broadcaster for odom -> base_link
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to cmd_vel to receive commands
        self.command_sub = self.create_subscription(Twist, "cmd_vel", self.cmd_vel_callback, 1)

        # Subscribe to cmd_pos for position commands
        self.position_sub = self.create_subscription(PointStamped, "cmd_pos", self.cmd_pos_callback, 1)

        # Subscribe to odometry for position feedback
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 1)

        # Create timer for continuous control loop
        timer_period = 1.0 / self.control_rate_hz
        self.control_timer = self.create_timer(timer_period, self.control_timer_callback)

        # Create timer for periodic state broadcasting (10Hz)
        state_timer_period = 0.1  # 10Hz
        self.state_timer = self.create_timer(state_timer_period, self.state_timer_callback)

        # Create landing and takeoff services
        self.landing_srv = self.create_service(Trigger, 'land', self.land_srv_handler)
        self.takeoff_srv = self.create_service(Trigger, 'takeoff', self.takeoff_srv_handler)

        # Create scan/rotation service
        self.scan_srv = self.create_service(Trigger, 'scan_surround', self.scan_surround_handler)

        # Create Kalman filter reset service
        self.reset_kalman_srv = self.create_service(Trigger, 'reset_kalman', self.reset_kalman_srv_handler)

        # Publish initial UNINITIALIZED state
        self.publish_state()
        self.get_logger().info('Publishers, subscribers, and services created')

        # Now attempt to connect to Crazyflie
        self.scf = SyncCrazyflie(address, cf=Crazyflie(rw_cache=os.path.join(
            os.path.dirname(__file__),
            "cache"
        )))

        self.get_logger().info(f'Attempting to connect to Crazyflie at {address}...')
        try:
            self.scf.open_link()
            self.get_logger().info('Successfully connected to Crazyflie!')
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Crazyflie: {e}")
            self.get_logger().error(f"Node will continue running in UNINITIALIZED state")
            self.get_logger().error(f"Please check: 1) Crazyflie is powered on, 2) Radio address is correct, 3) Crazyradio PA is plugged in")
            # Don't return - let node continue in UNINITIALIZED state
            return

        # Connection successful - check decks
        checks = self.decks_check([
            ("deck", "bcMultiranger"),
            ("deck", "bcFlow2")
        ])
        if not checks[0]:
            self.get_logger().error("Multiranger not attached")
        if not checks[1]:
            self.get_logger().error("Flowdeck not attached")

        # Setup state logging (requires successful connection)
        self.setup_state_logging()

        # Connection successful - transition to LANDED state
        with self.state_lock:
            self.state = STATE_LANDED
        self.publish_state()
        self.get_logger().info('Crazyflie initialized successfully - state: LANDED')

    def setup_state_logging(self):
        """
        Setup state logging for Crazyflie telemetry.
        Publishes:
          - Odometry on 'odom' and TF (odom -> base_link)
          - Range topics on 'range/*' and 4-beam LaserScan on 'scan'
        """
        # Create logging configurations
        self.lg_stab1 = self.lg_stab_config1()
        self.lg_stab2 = self.lg_stab_config2()

        # Add configurations to Crazyflie
        self.scf.cf.log.add_config(self.lg_stab1)
        self.scf.cf.log.add_config(self.lg_stab2)

        # Start logging
        self.lg_stab1.start()
        self.lg_stab2.start()

        self.get_logger().info("State logging enabled")

    def decks_check(self, groups_names):
        num_decks = len(groups_names)
        events = [threading.Event() for i in range(num_decks)]
        decks_attached = [False] * num_decks

        def callback(i, name, value_str):
            self.get_logger().info(f"{name} {value_str}")
            if int(value_str):
                decks_attached[i] = True
            events[i].set()

        for i, (group, name) in enumerate(groups_names):
            self.scf.cf.param.add_update_callback(
                group=group, name=name, cb=functools.partial(callback, i))

        for i, event, (group, name) in zip(range(num_decks), events, groups_names):
            event.wait()
            self.scf.cf.param.remove_update_callback(group=group, name=name)

        return decks_attached

    def cmd_vel_callback(self, data):
        """Store incoming cmd_vel for one-shot execution in next control cycle"""
        with self.cmd_lock:
            self.cmd_buffer = data
        # Switch to velocity mode when external velocity command received
        with self.position_lock:
            self.control_mode = CONTROL_MODE_VELOCITY
            self.position_target = None
            self.position_reached = False

    def cmd_pos_callback(self, data):
        """Store incoming position target"""
        with self.position_lock:
            self.position_target = data
            self.control_mode = CONTROL_MODE_POSITION
            self.position_reached = False  # Reset when new target received

    def odom_callback(self, data):
        """Store current odometry for position feedback"""
        with self.odom_lock:
            self.current_odom = data

    def publish_state(self):
        """Publish current state to crazyflie_state topic"""
        with self.state_lock:
            current_state = self.state

        state_msg = Int8()
        state_msg.data = current_state
        self.state_pub.publish(state_msg)

    def land_srv_handler(self, request, response):
        """Landing service: gradual descent then send stop_setpoint"""
        with self.state_lock:
            self.state = STATE_LANDING

        self.publish_state()  # Broadcast state change
        self.get_logger().info("Landing initiated...")

        # Gradual descent over 0.5 seconds (5 steps × 0.1s)
        for i in range(5):
            height = self.height - 0.1 * i
            self.scf.cf.commander.send_hover_setpoint(0, 0, 0, height)
            time.sleep(0.2)
        time.sleep(0.4)

        # Send stop setpoint
        self.scf.cf.commander.send_stop_setpoint()

        with self.state_lock:
            self.state = STATE_LANDED

        self.publish_state()  # Broadcast state change
        self.get_logger().info("Landing complete")
        response.success = True
        response.message = "Landed successfully"
        return response

    def takeoff_srv_handler(self, request, response):
        """Takeoff service: gradual ascent from landed state"""
        with self.state_lock:
            if self.state != STATE_LANDED:
                response.success = False
                response.message = f"Cannot takeoff: not in LANDED state (current state: {self.state})"
                return response
            self.state = STATE_TAKING_OFF

        self.publish_state()  # Broadcast state change
        self.get_logger().info("Takeoff initiated...")

        # Reset Kalman filter before takeoff
        self.get_logger().info("Resetting Kalman filter before takeoff...")
        try:
            self.scf.cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            self.scf.cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(0.1)
            self.get_logger().info("Kalman filter reset complete")
        except Exception as e:
            self.get_logger().warn(f"Kalman filter reset failed: {e}")

        # Gradual ascent over 0.5 seconds (5 steps × 0.1s)
        for i in range(5):
            height = 0.1 + 0.1 * i  # 0.1, 0.2, 0.3, 0.4, 0.5
            self.scf.cf.commander.send_hover_setpoint(0, 0, 0, height)
            time.sleep(0.2)

        with self.state_lock:
            self.state = STATE_FLYING

        self.publish_state()  # Broadcast state change
        self.get_logger().info("Takeoff complete")
        response.success = True
        response.message = "Takeoff successful"
        return response

    def scan_surround_handler(self, request, response):
        """Scan surroundings: rotate 90 degrees in place"""
        with self.state_lock:
            current_state = self.state

        if current_state != STATE_FLYING:
            response.success = False
            response.message = f"Cannot scan: not in FLYING state (current state: {current_state})"
            return response

        self.get_logger().info("Scan surround: rotating 90 degrees...")

        # Calculate rotation time: 90 degrees / angular_speed_factor (deg/s)
        rotation_time = 90.0 / self.angular_speed_factor

        # Send angular velocity command
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.angular.z = math.radians(self.angular_speed_factor)  # Full rotation speed
        self.cmd_vel_callback(twist)

        # Wait for rotation to complete
        time.sleep(rotation_time)

        # Stop rotation
        twist.angular.z = 0.0
        self.cmd_vel_callback(twist)

        # Small delay to stabilize
        time.sleep(0.2)

        self.get_logger().info("Scan complete")
        response.success = True
        response.message = "90 degree rotation complete"
        return response

    def reset_kalman_srv_handler(self, request, response):
        """Reset Kalman filter estimator"""
        self.get_logger().info("Resetting Kalman filter...")

        try:
            # Reset the Kalman estimator using the standard cflib method
            self.scf.cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            self.scf.cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(0.1)

            self.get_logger().info("Kalman filter reset complete")
            response.success = True
            response.message = "Kalman filter reset successfully"
        except Exception as e:
            self.get_logger().error(f"Failed to reset Kalman filter: {e}")
            response.success = False
            response.message = f"Kalman filter reset failed: {e}"

        return response

    def process_and_send_velocity(self, vx, vy, angular_z):
        """
        Process velocity commands, apply scaling/capping, and send to drone.
        Only scales down if magnitude exceeds speed_factor, otherwise passes through.
        """
        # Scale linear velocity only if magnitude exceeds speed_factor
        linear_magnitude = math.sqrt(vx**2 + vy**2)
        if linear_magnitude > self.linear_speed_factor:
            # Scale down to speed_factor
            vx_scaled = (vx / linear_magnitude) * self.linear_speed_factor
            vy_scaled = (vy / linear_magnitude) * self.linear_speed_factor
        else:
            # Pass through unchanged
            vx_scaled = vx
            vy_scaled = vy

        # Convert angular velocity to degrees and cap at angular_speed_factor
        angular_deg = math.degrees(angular_z)
        if abs(angular_deg) > self.angular_speed_factor:
            angular_deg = math.copysign(self.angular_speed_factor, angular_deg)

        # Publish setpoint
        setpoint_msg = Twist()
        setpoint_msg.linear.x = vx_scaled
        setpoint_msg.linear.y = vy_scaled
        setpoint_msg.linear.z = self.height
        setpoint_msg.angular.z = angular_deg
        self.setpoint_pub.publish(setpoint_msg)

        # Send command to drone
        self.scf.cf.commander.send_hover_setpoint(
            vx_scaled, vy_scaled, angular_deg, self.height
        )

    def control_timer_callback(self):
        """Continuous control loop running at control_rate_hz"""
        # Check current state
        with self.state_lock:
            current_state = self.state

        # Handle state-specific behavior
        if current_state == STATE_UNINITIALIZED:
            # Crazyflie not connected - do nothing
            return

        elif current_state == STATE_LANDED:
            # Keep sending stop_setpoint to maintain communication
            self.scf.cf.commander.send_stop_setpoint()
            return

        elif current_state in [STATE_LANDING, STATE_TAKING_OFF]:
            # Service is handling the transition, skip
            return

        elif current_state == STATE_FLYING:
            # Check control mode
            with self.position_lock:
                mode = self.control_mode
                target_pos = self.position_target

            if mode == CONTROL_MODE_VELOCITY:
                # Velocity control mode (original behavior)
                # Read and clear command buffer (one-shot execution)
                with self.cmd_lock:
                    cmd = self.cmd_buffer
                    # Clear buffer immediately after reading
                    self.cmd_buffer = Twist()

                # Get input velocities
                vx = cmd.linear.x
                vy = cmd.linear.y
                angular = cmd.angular.z

                # Process and send velocity
                self.process_and_send_velocity(vx, vy, angular)

            elif mode == CONTROL_MODE_POSITION:
                # Position control mode
                if target_pos is None:
                    # No target, hover in place
                    self.process_and_send_velocity(0.0, 0.0, 0.0)
                    return

                # Get current odometry
                with self.odom_lock:
                    current_odom = self.current_odom

                if current_odom is None:
                    # No odometry available yet, hover in place
                    self.process_and_send_velocity(0.0, 0.0, 0.0)
                    return

                # Calculate position error in world frame
                current_x = current_odom.pose.pose.position.x
                current_y = current_odom.pose.pose.position.y
                current_z = current_odom.pose.pose.position.z

                target_x = target_pos.point.x
                target_y = target_pos.point.y
                target_z = target_pos.point.z if target_pos.point.z > 0 else self.height

                error_x = target_x - current_x
                error_y = target_y - current_y
                distance = math.sqrt(error_x**2 + error_y**2)

                # Extract current yaw from quaternion
                q = current_odom.pose.pose.orientation
                # yaw = atan2(2(qw*qz + qx*qy), 1 - 2(qy^2 + qz^2))
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y**2 + q.z**2))

                if distance > self.position_tolerance:
                    # Far from target: generate velocity commands
                    # Calculate velocity direction (normalized)
                    direction_x = error_x / distance
                    direction_y = error_y / distance

                    # Scale to max speed
                    vel_world_x = direction_x * self.linear_speed_factor
                    vel_world_y = direction_y * self.linear_speed_factor

                    # Transform from world frame to body frame
                    cos_yaw = math.cos(yaw)
                    sin_yaw = math.sin(yaw)
                    vel_body_x = cos_yaw * vel_world_x + sin_yaw * vel_world_y
                    vel_body_y = -sin_yaw * vel_world_x + cos_yaw * vel_world_y

                    # Send velocity command
                    self.process_and_send_velocity(vel_body_x, vel_body_y, 0.0)

                    # Publish false if we were previously within tolerance
                    if self.position_reached:
                        msg = Bool()
                        msg.data = False
                        self.position_reached_pub.publish(msg)
                        self.position_reached = False

                else:
                    # Close to target: use position setpoint for precise holding
                    yaw_degrees = math.degrees(yaw)
                    self.scf.cf.commander.send_position_setpoint(
                        target_x, target_y, target_z, yaw_degrees
                    )

                    # Publish true when we first reach tolerance
                    if not self.position_reached:
                        msg = Bool()
                        msg.data = True
                        self.position_reached_pub.publish(msg)
                        self.position_reached = True
                        self.get_logger().info(f"Position target reached: ({target_x:.2f}, {target_y:.2f})")

    def state_timer_callback(self):
        """Periodic callback to broadcast current state at 10Hz"""
        self.publish_state()

    def lg_stab_config1(self):
        """Position/orientation logging"""
        lg = LogConfig(name='Position', period_in_ms=100)
        lg.add_variable('stateEstimate.x', 'float')
        lg.add_variable('stateEstimate.y', 'float')
        lg.add_variable('stateEstimate.z', 'float')
        lg.add_variable('stabilizer.roll', 'float')
        lg.add_variable('stabilizer.pitch', 'float')
        lg.add_variable('stabilizer.yaw', 'float')
        lg.data_received_cb.add_callback(self.log_stab_callback1)
        return lg

    def lg_stab_config2(self):
        """Range sensors logging"""
        lg = LogConfig(name='Range', period_in_ms=100)
        lg.add_variable('range.front', 'uint16_t')
        lg.add_variable('range.back', 'uint16_t')
        lg.add_variable('range.left', 'uint16_t')
        lg.add_variable('range.right', 'uint16_t')
        lg.add_variable('range.up', 'uint16_t')
        lg.add_variable('range.zrange', 'uint16_t')
        lg.data_received_cb.add_callback(self.log_stab_callback2)
        return lg

    def _euler_to_quaternion(self, roll, pitch, yaw):
        """Convert radians roll/pitch/yaw to quaternion (x, y, z, w)."""
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * cp * cy
        return qx, qy, qz, qw
 
    def log_stab_callback1(self, timestamp, data, logconf):
        """Publish Odometry and TF (odom -> base_link), consistent with odometry_node.py."""
        stamp = self.get_clock().now().to_msg()
        # Estimator positions in meters, stabilizer angles in degrees
        x = float(data['stateEstimate.x'])
        y = float(data['stateEstimate.y'])
        z = float(data['stateEstimate.z'])
        roll = math.radians(float(data['stabilizer.roll']))
        pitch = math.radians(float(data['stabilizer.pitch']))
        yaw = math.radians(float(data['stabilizer.yaw']))
        qx, qy, qz, qw = self._euler_to_quaternion(roll, pitch, yaw)
 
        # Odometry message
        odom = Odometry()
        odom.header = Header(stamp=stamp, frame_id=self.odom_frame)
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self.odom_pub.publish(odom)
 
        # TF: odom -> base_link
        t = TransformStamped()
        t.header = Header(stamp=stamp, frame_id=self.odom_frame)
        t.child_frame_id = self.base_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

    def log_stab_callback2(self, timestamp, data, logconf):
        """Publish individual Range topics and a 4-beam LaserScan, consistent with ranger_node.py."""
        stamp = self.get_clock().now().to_msg()
 
        # Raw mm from FW -> meters
        def mm_to_m(v):
            try:
                return float(v) / 1000.0
            except Exception:
                return float('inf')
 
        ranges = {
            'right': mm_to_m(data.get('range.right')),
            'front': mm_to_m(data.get('range.front')),
            'left':  mm_to_m(data.get('range.left')),
            'back':  mm_to_m(data.get('range.back')),
            'up':    mm_to_m(data.get('range.up')),
        }
 
        # Publish Range messages
        for direction, dist in ranges.items():
            if direction not in self.range_pubs:
                continue
            r = Range()
            r.header = Header(stamp=stamp, frame_id=f'{self.base_frame}/{direction}')
            r.radiation_type = Range.INFRARED
            r.field_of_view = self.fov
            r.min_range = self.min_range
            r.max_range = self.max_range
            # clamp to sensor limits (LaserScan helper downstream expects finite values)
            if math.isfinite(dist):
                r.range = max(self.min_range, min(self.max_range, dist))
            else:
                r.range = float('inf')
            self.range_pubs[direction].publish(r)
 
        # Publish LaserScan over 4 horizontal sensors in order: [back, right, front, left]
        scan = LaserScan()
        scan.header = Header(stamp=stamp, frame_id=self.base_frame)
        # Exactly 4 beams -> angle_max = angle_min + 3 * angle_increment
        scan.angle_min = 0.0
        scan.angle_increment = math.pi / 2.0
        scan.angle_max = scan.angle_min + 3.0 * scan.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = self.min_range
        scan.range_max = self.max_range
        def clamp(d):
            if not math.isfinite(d) or d > self.max_range:
                return float('inf')
            return max(self.min_range, d)
        scan.ranges = [
            clamp(ranges['back']),
            clamp(ranges['right']),
            clamp(ranges['front']),
            clamp(ranges['left']),
        ]
        self.scan_pub.publish(scan)

    def dispose(self):
        if self.scf._is_link_open:
            # Stop state logging if it was enabled
            if hasattr(self, 'lg_stab1') and self.lg_stab1:
                self.lg_stab1.stop()
            if hasattr(self, 'lg_stab2') and self.lg_stab2:
                self.lg_stab2.stop()
            self.scf.close_link()


def main(args=None):
    rclpy.init(args=args)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    time.sleep(1)

    handle = CrazyflieCommunicator()

    try:
        rclpy.spin(handle)
    except KeyboardInterrupt:
        pass
    finally:
        handle.dispose()
        handle.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
