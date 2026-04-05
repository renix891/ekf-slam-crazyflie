#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('gslam_ros2')

    # Path to config files
    mission_config = PathJoinSubstitution([pkg_share, 'config', 'mission.yaml'])

    # Declare launch arguments
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='0.5',
        description='Target flight height in meters'
    )

    address_arg = DeclareLaunchArgument(
        'address',
        default_value='radio://0/80/2M/E7E7E7E7E2',
        description='Crazyflie radio address'
    )

    linear_speed_factor_arg = DeclareLaunchArgument(
        'linear_speed_factor',
        default_value='0.5',
        description='Linear speed scaling factor'
    )

    angular_speed_factor_arg = DeclareLaunchArgument(
        'angular_speed_factor',
        default_value='90.0',
        description='Angular speed scaling factor (degrees/sec)'
    )

    control_rate_hz_arg = DeclareLaunchArgument(
        'control_rate_hz',
        default_value='30.0',
        description='Control loop frequency in Hz'
    )

    position_tolerance_arg = DeclareLaunchArgument(
        'position_tolerance',
        default_value='0.05',
        description='Position tolerance for switching to position hold mode (meters)'
    )

    odom_frame_arg = DeclareLaunchArgument(
        'odom_frame',
        default_value='odom',
        description='Odometry frame name'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame name'
    )

    max_waypoint_distance_arg = DeclareLaunchArgument(
        'max_waypoint_distance',
        default_value='1.0',
        description='Maximum distance to next waypoint (meters)'
    )

    nav_enabled_arg = DeclareLaunchArgument(
        'nav_enabled',
        default_value='false',
        description='Enable autonomous navigation'
    )

    enable_monitor_arg = DeclareLaunchArgument(
        'enable_monitor',
        default_value='false',
        description='Enable range monitor visualization'
    )

    # Controller node (now handles odometry and range publishing internally)
    controller_node = Node(
        package='gslam_ros2',
        executable='controller',
        name='crazyflie_controller',
        output='screen',
        parameters=[{
            'height': LaunchConfiguration('height'),
            'address': LaunchConfiguration('address'),
            'linear_speed_factor': LaunchConfiguration('linear_speed_factor'),
            'angular_speed_factor': LaunchConfiguration('angular_speed_factor'),
            'control_rate_hz': LaunchConfiguration('control_rate_hz'),
            'position_tolerance': LaunchConfiguration('position_tolerance'),
            'odom_frame': LaunchConfiguration('odom_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
        }]
    )

    # 2D Mapper node
    mapper_2d_node = Node(
        package='gslam_ros2',
        executable='mapper_2d',
        name='mapper_2d',
        output='screen',
        parameters=[{
            'map_size_x': 20.0,
            'map_size_y': 20.0,
            'map_origin_x': -10.0,
            'map_origin_y': -10.0,
            'map_resolution': 0.1,
        }]
    )

    # D* Lite Planner node
    planner_node = Node(
        package='gslam_ros2',
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=[{
            'flight_height': LaunchConfiguration('height'),
            'waypoint_skip_distance': 0.3,
        }]
    )

    # Navigation node
    nav_node = Node(
        package='gslam_ros2',
        executable='nav_node',
        name='nav_node',
        output='screen',
        parameters=[{
            'max_waypoint_distance': LaunchConfiguration('max_waypoint_distance'),
            'enabled': LaunchConfiguration('nav_enabled'),
            'flight_height': LaunchConfiguration('height'),
        }]
    )

    # Mission Manager node
    mission_manager_node = Node(
        package='gslam_ros2',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[mission_config]
    )

    # Edge Detector node
    edge_detector_node = Node(
        package='gslam_ros2',
        executable='edge_detector_node',
        name='edge_detector_node',
        output='screen',
        parameters=[mission_config]
    )

    # Range Monitor node (optional)
    range_monitor_node = Node(
        package='gslam_ros2',
        executable='range_monitor_node',
        name='range_monitor_node',
        output='screen',
        parameters=[mission_config],
        condition=IfCondition(LaunchConfiguration('enable_monitor'))
    )

    return LaunchDescription([
        height_arg,
        address_arg,
        linear_speed_factor_arg,
        angular_speed_factor_arg,
        control_rate_hz_arg,
        position_tolerance_arg,
        odom_frame_arg,
        base_frame_arg,
        max_waypoint_distance_arg,
        nav_enabled_arg,
        enable_monitor_arg,
        controller_node,
        mapper_2d_node,
        planner_node,
        nav_node,
        mission_manager_node,
        edge_detector_node,
        range_monitor_node,
    ])
