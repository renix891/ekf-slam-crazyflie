"""Launch file for Crazyflie node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Crazyflie node."""

    # Declare launch arguments
    uri_arg = DeclareLaunchArgument(
        'uri',
        default_value='radio://0/80/2M/E7E7E7E7E2',
        description='Crazyflie URI'
    )

    hover_height_arg = DeclareLaunchArgument(
        'hover_height',
        default_value='0.3',
        description='Default hover height in meters'
    )

    speed_factor_arg = DeclareLaunchArgument(
        'speed_factor',
        default_value='0.3',
        description='Speed factor for velocity commands'
    )

    resolution_arg = DeclareLaunchArgument(
        'octomap_resolution',
        default_value='0.05',
        description='OctoMap voxel resolution in meters'
    )

    max_range_arg = DeclareLaunchArgument(
        'max_range',
        default_value='3.0',
        description='Maximum sensor range to trust in meters'
    )

    # Crazyflie node
    crazyflie_node = Node(
        package='crazyflie_ros2',
        executable='crazyflie_node',
        name='crazyflie_node',
        output='screen',
        parameters=[{
            'uri': LaunchConfiguration('uri'),
            'hover_height': LaunchConfiguration('hover_height'),
            'speed_factor': LaunchConfiguration('speed_factor'),
        }]
    )

    # OctoMap mapper node
    octomap_mapper_node = Node(
        package='crazyflie_ros2',
        executable='octomap_mapper_node',
        name='octomap_mapper_node',
        output='screen',
        parameters=[{
            'resolution': LaunchConfiguration('octomap_resolution'),
            'max_range': LaunchConfiguration('max_range'),
            'min_range': 0.02,
            'world_frame': 'world',
            'robot_frame': 'crazyflie',
            'publish_point_cloud': True,
            'point_cloud_history': 100,
        }]
    )

    return LaunchDescription([
        uri_arg,
        hover_height_arg,
        speed_factor_arg,
        resolution_arg,
        max_range_arg,
        crazyflie_node,
        octomap_mapper_node,
    ])
