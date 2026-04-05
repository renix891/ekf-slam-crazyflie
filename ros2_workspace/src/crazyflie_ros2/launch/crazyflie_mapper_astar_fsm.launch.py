from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')

    def setup(context, *args, **kwargs):
        log_level = LaunchConfiguration('log_level').perform(context)

        crazyflie_node = Node(
            package='crazyflie_ros2',
            executable='crazyflie_node',
            name='crazyflie_node',
            output='screen',
            arguments=['--ros-args', '--log-level', f'crazyflie_node:={log_level}'],
            parameters=[{
                'uri': 'radio://0/80/2M/E7E7E7E7E2',
                'hover_height': 0.5,
                'speed_factor': 0.3,
                'logging_only': False,
                'map_frame': 'map',
                'world_frame': 'world',
                'odom_frame': 'crazyflie/odom',
                'base_frame': 'crazyflie',
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

        astar_path_planning_node = Node(
            package='crazyflie_ros2',
            executable='astar_path_planning_node',
            name='astar_path_planning_node',
            output='screen',
            arguments=['--ros-args', '--log-level', 'astar_path_planning_node:=info'],
            parameters=[{
                'planning_frequency': 2.0,
                'flight_height': 0.5,
            }]
        )

        autonomous_navigation_node = Node(
            package='crazyflie_ros2',
            executable='autonomous_navigation_node',
            name='autonomous_navigation_node',
            output='screen',
            arguments=['--ros-args', '--log-level', log_level],
            parameters=[{
                'waypoint_tolerance': 0.2,
                'scanning_yaw_rate': 1.6,
                'flight_height': 0.5,
                'active_scanning.enabled': True,
                'active_scanning.speed_scale': 0.7,
                'active_scanning.lookahead_tau': 0.2,
                'active_scanning.vz_kp': 1.0,
            }]
        )

        edge_detector_node = Node(
            package='crazyflie_ros2',
            executable='range_edge_detector_node',
            name='range_edge_detector_node',
            output='screen',
            arguments=['--ros-args', '--log-level', f'range_edge_detector_node:={log_level}'],
            parameters=[{
                'topic': '/crazyflie/range/down',
                'window': 20,
                'z_thresh': 3.8,
                'expected_height_m': 0.3,
                'log_csv': True,
                'log_dir': 'logs'
            }]
        )

        return [
            crazyflie_node,
            simple_mapper_node,
            astar_path_planning_node,
            autonomous_navigation_node,
            edge_detector_node
        ]

    return LaunchDescription([log_level_arg, OpaqueFunction(function=setup)])
