"""Gazebo Harmonic + ros_gz_bridge + EKF-SLAM stack for simulation testing."""

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


# The simulation repo lives alongside ros2_workspace under the project root.
# Using an absolute path avoids brittle ../ counts that differ between the
# source tree and the installed launch location.
PROJECT_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
SIM_GAZEBO_DIR = os.path.join(
    PROJECT_DIR, 'simulation_ws', 'crazyflie-simulation',
    'simulator_files', 'gazebo')
WORLD_FILE = os.path.join(SIM_GAZEBO_DIR, 'worlds', 'crazyflie_world.sdf')
TAKEOFF_SCRIPT = os.path.join(
    PROJECT_DIR, 'ros2_workspace', 'src', 'ekf_slam', 'scripts',
    'gz_takeoff.py')
BAG_DIR = os.path.join(PROJECT_DIR, 'results', 'gazebo_test_bag')
BAG_TOPICS = ['/crazyflie/odom', '/crazyflie/scan', '/ekf_pose', '/map']


def generate_launch_description():
    bridge_config = os.path.join(
        get_package_share_directory('ekf_slam'), 'config', 'gz_bridge.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock from Gazebo for all ROS nodes.')

    # Let Gazebo find the `crazyflie` model via <include><uri>model://crazyflie</uri>.
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

    ekf_slam = Node(
        package='ekf_slam',
        executable='ekf_slam_node',
        name='ekf_slam_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    mapper = Node(
        package='crazyflie_mapper',
        executable='mapper_node',
        name='crazyflie_mapper_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen')

    # Scripted flight plan: arms motors, takes off, flies the sequence, lands.
    # Replaces the separate one-shot enable publisher.
    takeoff = ExecuteProcess(
        cmd=['python3', TAKEOFF_SCRIPT],
        output='screen')

    # Record the key topics to a ros2 bag for offline analysis.
    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', BAG_DIR] + BAG_TOPICS,
        output='screen')

    # `ros2 bag record -o` refuses to overwrite an existing directory.
    # Clean any previous run before recording starts.
    clean_bag_dir = ExecuteProcess(
        cmd=['rm', '-rf', BAG_DIR],
        output='screen')

    return LaunchDescription([
        declare_sim_time,
        set_resource_path,
        clean_bag_dir,
        gz_sim,
        TimerAction(period=3.0, actions=[bridge]),
        TimerAction(period=5.0, actions=[ekf_slam, mapper]),
        TimerAction(period=5.0, actions=[bag_record]),
        TimerAction(period=3.0, actions=[takeoff]),
    ])
