"""Full autonomous-navigation stack in Gazebo using EKF-corrected pose.

Pipeline tested:
    Gazebo -> bridge -> mapper + ekf_slam -> planning -> navigation -> box_landing
A goal_pose at the landing pad is published 5 s after launch so D* Lite plans
straight away. A bag captures odom, ekf_pose, map, planned_path, cmd_vel for
post-flight comparison against the odom-only baseline.
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
BAG_DIR = os.path.join(PROJECT_DIR, 'results', 'gazebo_full_nav_ekf_bag')
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

    ekf_slam = Node(
        package='ekf_slam',
        executable='ekf_slam_node',
        name='ekf_slam_node',
        parameters=[{'use_sim_time': use_sim_time}],
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
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    box_landing = Node(
        package='crazyflie_navigation',
        executable='box_landing_node',
        name='box_landing_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Climb to hover altitude and exit — releases /cmd_vel so navigation_node
    # can drive without contention.
    hover = ExecuteProcess(
        cmd=['python3', HOVER_SCRIPT],
        output='screen')

    # Enable the autonomous navigation service so the planner's path is followed.
    enable_nav = ExecuteProcess(
        cmd=[
            'ros2', 'service', 'call', '/enable_autonomous',
            'std_srvs/srv/SetBool', '{data: true}',
        ],
        output='screen')

    # Publish a single goal_pose so D* Lite plans toward the landing pad.
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

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', BAG_DIR] + BAG_TOPICS,
        output='screen')

    # ros2 bag refuses to overwrite; clean any previous run.
    clean_bag_dir = ExecuteProcess(
        cmd=['rm', '-rf', BAG_DIR],
        output='screen')

    return LaunchDescription([
        declare_sim_time,
        set_resource_path,
        clean_bag_dir,
        gz_sim,
        TimerAction(period=3.0, actions=[bridge, hover]),
        TimerAction(period=5.0, actions=[mapper, ekf_slam, bag_record]),
        TimerAction(period=6.0, actions=[planning, navigation, box_landing]),
        # Goal + nav-enable fire at t=12 so hover (~5 s) has fully released
        # /cmd_vel and EKF/mapper have ~7 s to seed pose and the map.
        TimerAction(period=12.0, actions=[publish_goal, enable_nav]),
    ])
