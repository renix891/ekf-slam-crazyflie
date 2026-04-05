"""Launch file for Crazyflie with Simple 2D Mapper."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Crazyflie with Simple 2D Mapper integration."""

    # Declare launch arguments
    uri_arg = DeclareLaunchArgument(
        'uri',
        default_value='radio://0/80/2M/E7E7E7E7E2',
        description='Crazyflie URI'
    )

    hover_height_arg = DeclareLaunchArgument(
        'hover_height',
        default_value='0.5',
        description='Default hover height in meters'
    )

    speed_factor_arg = DeclareLaunchArgument(
        'speed_factor',
        default_value='0.3',
        description='Speed factor for velocity commands'
    )

    logging_only_arg = DeclareLaunchArgument(
        'logging_only',
        default_value='false',
        description='Enable logging-only mode (no flight commands sent to drone)'
    )

    robot_prefix_arg = DeclareLaunchArgument(
        'robot_prefix',
        default_value='/crazyflie',
        description='Robot namespace prefix'
    )

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Logging level (debug, info, warn, error, fatal)'
    )

    # Crazyflie node
    crazyflie_node = Node(
        package='crazyflie_ros2',
        executable='crazyflie_node',
        name='crazyflie_node',
        output='screen',
        arguments=['--ros-args', '--log-level', ['crazyflie_node:=', LaunchConfiguration('log_level')]],
        parameters=[{
            'uri': LaunchConfiguration('uri'),
            'hover_height': LaunchConfiguration('hover_height'),
            'speed_factor': LaunchConfiguration('speed_factor'),
            'logging_only': LaunchConfiguration('logging_only'),
        }]
    )

    # Simple Mapper node (publishes 2D occupancy grid from multiranger sensors)
    simple_mapper_node = Node(
        package='crazyflie_ros2',
        executable='simple_mapper_node',
        name='simple_mapper_node',
        output='screen',
        arguments=['--ros-args', '--log-level', ['simple_mapper_node:=', LaunchConfiguration('log_level')]],
        parameters=[{
            'robot_prefix': LaunchConfiguration('robot_prefix'),
            'use_bayesian_updates': True,  # Enable Bayesian occupancy updates
            'avoidance_distance': 0.6,     # meters (increased by 0.1m)
            'max_avoidance_weight': 50,    # 1-50 range
            # Map configuration (Option B: extends in +X direction)
            'map_size_x': 40.0,           # meters (40m x 20m map)
            'map_size_y': 20.0,           # meters
            'map_origin_x': -10.0,        # X range: -10 to +30m
            'map_origin_y': -10.0,        # Y range: -10 to +10m
            'map_resolution': 0.1,        # meters per cell
        }]
    )

    # Autonomous Navigation node (path planning and navigation)
    autonomous_navigation_node = Node(
        package='crazyflie_ros2',
        executable='autonomous_navigation_node',
        name='autonomous_navigation_node',
        output='screen',
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        parameters=[{
            'waypoint_tolerance': 0.2,
            'planning_frequency': 2.0,
            'scanning_yaw_rate': 0.5,
            'flight_height': 0.3,
            'scanning_yaw_rate': 1.0,                 # faster spin
            'active_scanning.speed_scale': 0.5,       # slower linear motion
        }]
    )

    return LaunchDescription([
        uri_arg,
        hover_height_arg,
        speed_factor_arg,
        logging_only_arg,
        robot_prefix_arg,
        log_level_arg,
        crazyflie_node,
        simple_mapper_node,
        autonomous_navigation_node,
    ])
