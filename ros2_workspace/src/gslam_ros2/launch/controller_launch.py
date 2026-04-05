from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('gslam_ros2')

    # Path to config files
    swarm_config = PathJoinSubstitution([pkg_share, 'config', 'swarm.yaml'])

    # Controller node
    controller_node = Node(
        package='gslam_ros2',
        executable='controller.py',
        name='controller',
        output='screen',
        parameters=[swarm_config]
    )

    return LaunchDescription([
        controller_node,
    ])
