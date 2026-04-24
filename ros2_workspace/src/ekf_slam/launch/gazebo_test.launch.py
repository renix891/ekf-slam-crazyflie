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
SIM_GAZEBO_DIR = '/home/renix/EKF-SLAM-Autonomous-Crazyflie/simulation_ws/' \
    'crazyflie-simulation/simulator_files/gazebo'
WORLD_FILE = os.path.join(SIM_GAZEBO_DIR, 'worlds', 'crazyflie_world.sdf')


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

    # One-shot publisher: arm the multicopter velocity controller. Fires 3 s
    # after the bridge starts so the ROS→GZ /crazyflie/enable bridge is up.
    enable_motors = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--once',
             '/crazyflie/enable', 'std_msgs/msg/Bool', '{data: true}'],
        output='screen')

    return LaunchDescription([
        declare_sim_time,
        set_resource_path,
        gz_sim,
        TimerAction(period=3.0, actions=[bridge]),
        TimerAction(period=5.0, actions=[ekf_slam, mapper]),
        TimerAction(period=6.0, actions=[enable_motors]),
    ])
