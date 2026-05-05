"""Final validation experiment — odometry-only baseline run.

Byte-for-byte copy of gazebo_full_nav.launch.py with three changes:
  * ekf_slam_node replaced by odom_to_pose.py (publishes /ekf_pose from
    /crazyflie/odom + cumulative Brownian noise on x, y, yaw — simulates
    an IMU-only pose estimate without external correction).
  * BAG_DIR points at results/final_odom_bag/.
  * mission_orchestrator added at t=12s for the return leg.

Compare landing accuracy on the return leg vs the EKF run to demonstrate
the value of EKF-SLAM correction.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
SIM_GAZEBO_DIR = os.path.join(
    PROJECT_DIR, 'simulation_ws', 'crazyflie-simulation',
    'simulator_files', 'gazebo')
WORLD_FILE = os.path.join(SIM_GAZEBO_DIR, 'worlds', 'crazyflie_world.sdf')
HOVER_SCRIPT = os.path.join(
    PROJECT_DIR, 'ros2_workspace', 'src', 'ekf_slam', 'scripts',
    'gz_hover.py')
ODOM_TO_POSE_SCRIPT = os.path.join(
    PROJECT_DIR, 'ros2_workspace', 'src', 'ekf_slam', 'scripts',
    'odom_to_pose.py')
ORCHESTRATOR_SCRIPT = os.path.join(
    PROJECT_DIR, 'ros2_workspace', 'src', 'ekf_slam', 'scripts',
    'mission_orchestrator.py')
BAG_DIR = os.path.join(PROJECT_DIR, 'results', 'final_odom_bag')
BAG_TOPICS = [
    '/crazyflie/odom',
    '/crazyflie/scan',
    '/ekf_pose',
    '/map',
    '/planned_path',
    '/goal_pose',
    '/cmd_vel',
]

# Landing-pad goal (matches the box position assumed by the box_landing test).
GOAL_X = 0.8
GOAL_Y = 0.0
GOAL_Z = 0.3


def generate_launch_description():
    bridge_config = os.path.join(
        get_package_share_directory('ekf_slam'), 'config', 'gz_bridge.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock from Gazebo for all ROS nodes.')

    set_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=SIM_GAZEBO_DIR)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {WORLD_FILE}'}.items())

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': use_sim_time,
        }],
        output='screen')

    mapper = Node(
        package='crazyflie_mapper',
        executable='mapper_node',
        name='crazyflie_mapper_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Pose source: odom + Brownian noise. No EKF-SLAM node in this run.
    odom_to_pose = ExecuteProcess(
        cmd=['python3', ODOM_TO_POSE_SCRIPT,
             '--ros-args',
             '-p', 'enable_noise:=true',
             # Hardware-grounded noise: PMW3901 flow + VL53L1x ToF +
             # BMI088 gyro. See odom_to_pose.py for derivation.
             '-p', 'sigma_xy_per_s:=0.020',
             '-p', 'sigma_yaw_per_s:=0.001'],
        output='screen')

    planning = Node(
        package='crazyflie_planning',
        executable='planning_node',
        name='planning_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    navigation = Node(
        package='crazyflie_navigation',
        executable='navigation_node',
        name='autonomous_navigation_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            # Odom-only baseline: skip per-waypoint SCANNING rotations.
            # Scanning is an EKF-oriented behavior (rotate to feed the
            # observation step with multi-bearing landmark hits); for the
            # noisy-odom run we want pure point-to-point navigation so
            # the comparison isolates pose-estimation accuracy.
            'scanning_enabled': False,
            # Halved from the 0.30 m/s default to keep bank angles within
            # the Gazebo MulticopterVelocityControl plugin's altitude-
            # tracking envelope. Without scan-spins between waypoints the
            # drone otherwise sustains lateral commands long enough that
            # the simulator drops altitude during the bank — a sim
            # artifact, not a controller reality. The EKF run keeps 0.30
            # because its periodic rotational scans give the controller
            # implicit altitude-recovery windows.
            'max_velocity': 0.15,
        }],
        output='screen')

    hover = ExecuteProcess(
        cmd=['python3', HOVER_SCRIPT],
        output='screen')

    enable_nav = ExecuteProcess(
        cmd=[
            'ros2', 'service', 'call', '/enable_autonomous',
            'std_srvs/srv/SetBool', '{data: true}',
        ],
        output='screen')

    publish_goal = ExecuteProcess(
        cmd=[
            'ros2', 'topic', 'pub', '--once', '/goal_pose',
            'geometry_msgs/msg/PoseStamped',
            (
                "{header: {frame_id: 'map'}, "
                f"pose: {{position: {{x: {GOAL_X}, y: {GOAL_Y}, z: {GOAL_Z}}}, "
                "orientation: {w: 1.0}}}"
            ),
        ],
        output='screen')

    orchestrator = ExecuteProcess(
        cmd=['python3', ORCHESTRATOR_SCRIPT],
        output='screen')

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', BAG_DIR] + BAG_TOPICS,
        output='screen')

    clean_bag_dir = ExecuteProcess(
        cmd=['rm', '-rf', BAG_DIR],
        output='screen')

    return LaunchDescription([
        declare_sim_time,
        set_resource_path,
        clean_bag_dir,
        gz_sim,
        TimerAction(period=3.0, actions=[bridge, hover]),
        TimerAction(period=5.0, actions=[mapper, odom_to_pose, bag_record]),
        TimerAction(period=6.0, actions=[planning, navigation]),
        TimerAction(period=12.0, actions=[publish_goal, enable_nav, orchestrator]),
    ])
