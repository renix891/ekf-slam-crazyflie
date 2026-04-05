"""Launch file for Crazyflie SIMULATION with mapper, D* Lite, and FSM navigation using ABSOLUTE edge detection."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')

    # Planning parameters
    planning_frequency_arg = DeclareLaunchArgument(
        'planning_frequency', default_value='1',
        description='Path planning frequency in Hz (lower = less oscillation)')

    flight_height_arg = DeclareLaunchArgument(
        'flight_height', default_value='0.35',
        description='Flight height in meters')

    # Mapper parameters
    avoidance_distance_arg = DeclareLaunchArgument(
        'avoidance_distance', default_value='1.1',
        description='Avoidance distance from obstacles in meters (increased by 0.1m)')

    max_avoidance_weight_arg = DeclareLaunchArgument(
        'max_avoidance_weight', default_value='50',
        description='Maximum avoidance weight (1-50 range)')

    # Navigation parameters
    waypoint_tolerance_arg = DeclareLaunchArgument(
        'waypoint_tolerance', default_value='0.15',
        description='Waypoint tolerance in meters')

    scanning_yaw_rate_arg = DeclareLaunchArgument(
        'scanning_yaw_rate', default_value='2.0',
        description='Scanning yaw rate in rad/s')

    # Absolute edge detector parameters
    box_height_arg = DeclareLaunchArgument(
        'box_height', default_value='0.1',
        description='Expected box height in meters')

    edge_modifier_arg = DeclareLaunchArgument(
        'edge_modifier', default_value='0.5',
        description='Edge detection threshold modifier (0.0-1.0)')

    edge_detection_delay_arg = DeclareLaunchArgument(
        'edge_detection_delay', default_value='10.0',
        description='Delay before edge detection starts (seconds)')

    def setup(context, *args, **kwargs):
        log_level = LaunchConfiguration('log_level').perform(context)
        planning_frequency = float(LaunchConfiguration('planning_frequency').perform(context))
        flight_height = float(LaunchConfiguration('flight_height').perform(context))
        avoidance_distance = float(LaunchConfiguration('avoidance_distance').perform(context))
        max_avoidance_weight = int(LaunchConfiguration('max_avoidance_weight').perform(context))
        waypoint_tolerance = float(LaunchConfiguration('waypoint_tolerance').perform(context))
        scanning_yaw_rate = float(LaunchConfiguration('scanning_yaw_rate').perform(context))
        box_height = float(LaunchConfiguration('box_height').perform(context))
        edge_modifier = float(LaunchConfiguration('edge_modifier').perform(context))
        edge_detection_delay = float(LaunchConfiguration('edge_detection_delay').perform(context))

        crazyflie_sim_node = Node(
            package='crazyflie_ros2',
            executable='crazyflie_sim_node',
            name='crazyflie_sim_node',
            output='screen',
            arguments=['--ros-args', '--log-level', f'crazyflie_sim_node:={log_level}'],
            parameters=[{
                'frame_id': 'crazyflie',
                'map_frame': 'map',
                'world_frame': 'world',
                'odom_frame': 'crazyflie/odom',
                'base_frame': 'crazyflie',
                'publish_rate_hz': 10.0,
                'takeoff_height': flight_height,
                'default_yaw_rate': 0.3,
                'speed_limit': 1.0,
                'yaw_rate_limit': 2.0,
                'box_half_xy': 3.75,  # 50% larger (7.5x7.5m)
                'box_height': 2.0,
                'responsiveness': 0.3,  # Slower response like real drone
            }]
        )

        simple_mapper_node = Node(
            package='crazyflie_ros2',
            executable='simple_mapper_node',
            name='simple_mapper_node',
            output='screen',
            arguments=['--ros-args', '--log-level', f'simple_mapper_node:={log_level}'],
            parameters=[{
                'robot_prefix': '/crazyflie',
                'use_bayesian_updates': True,  # Enable Bayesian occupancy updates
                'avoidance_distance': avoidance_distance,
                'max_avoidance_weight': max_avoidance_weight,
                # Map configuration (Option B: extends in +X direction)
                'map_size_x': 20.0,           # meters (20m x 20m map)
                'map_size_y': 20.0,           # meters
                'map_origin_x': -10.0,        # X range: -10 to +10m
                'map_origin_y': -10.0,        # Y range: -10 to +10m
                'map_resolution': 0.05,        # meters per cell
            }]
        )

        dstarlite_path_planning_node = Node(
            package='crazyflie_ros2',
            executable='dstarlite_path_planning_node',
            name='dstarlite_path_planning_node',
            output='screen',
            arguments=['--ros-args', '--log-level', f'dstarlite_path_planning_node:={log_level}'],
            parameters=[{
                'planning_frequency': planning_frequency,
                'flight_height': flight_height,
            }]
        )

        autonomous_navigation_node = Node(
            package='crazyflie_ros2',
            executable='autonomous_navigation_node',
            name='autonomous_navigation_node',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{
                'waypoint_tolerance': waypoint_tolerance,
                'scanning_yaw_rate': scanning_yaw_rate,
                'flight_height': flight_height,
                'active_scanning.enabled': True,
                'active_scanning.speed_scale': 0.7,
                'active_scanning.lookahead_tau': 0.2,
                'active_scanning.vz_kp': 1.0,
            }]
        )

        # ABSOLUTE EDGE DETECTOR (replaces z-score based detector)
        edge_detector_node = Node(
            package='crazyflie_ros2',
            executable='range_edge_detector_absolute_node',
            name='range_edge_detector_node',  # Keep same node name for compatibility
            output='screen',
            arguments=['--ros-args', '--log-level', f'range_edge_detector:={log_level}'],
            parameters=[{
                'topic': '/crazyflie/range/down',
                'flight_height': flight_height,
                'box_height': box_height,
                'modifier': edge_modifier,
                'edge_detection_delay': edge_detection_delay,
                'debounce_period': 0.3,
                'log_csv': True,
                'log_dir': 'logs'
            }]
        )

        mode_manager_node = Node(
            package='crazyflie_ros2',
            executable='mode_manager_node',
            name='mode_manager_node',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{
                'flight_height': flight_height,
                'altitude_tolerance': 0.05,
                'edge_detection_delay': 2.0,
                'landing_descent_rate': 0.01,
                'min_landing_height': 0.05,
            }]
        )

        box_landing_node = Node(
            package='crazyflie_ros2',
            executable='box_landing_node',
            name='box_landing_node',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{
                'enabled': False,
                'flight_height': flight_height,
                'search_span_m': 1.2,
                'search_step_m': 0.2,
                'max_descent_rate': 0.08,
                'min_hover_height': 0.12,
                'min_edge_points': 20,
                'goal_tolerance': 0.03,
                'box_landing_speed': 1.0,
                'search_scale': 4.0,
            }]
        )

        return [
            crazyflie_sim_node,
            simple_mapper_node,
            dstarlite_path_planning_node,
            autonomous_navigation_node,
            edge_detector_node,
            mode_manager_node,
            box_landing_node
        ]

    return LaunchDescription([
        log_level_arg,
        planning_frequency_arg,
        flight_height_arg,
        avoidance_distance_arg,
        max_avoidance_weight_arg,
        waypoint_tolerance_arg,
        scanning_yaw_rate_arg,
        box_height_arg,
        edge_modifier_arg,
        edge_detection_delay_arg,
        OpaqueFunction(function=setup)
    ])
