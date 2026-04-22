"""Full EKF-SLAM stack: hardware, mapper, EKF, planner, navigation, box landing."""

from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    crazyflie_hw = Node(
        package='crazyflie_ros2',
        executable='crazyflie_node',
        name='crazyflie_node',
        output='screen',
    )

    mapper = Node(
        package='crazyflie_mapper',
        executable='mapper_node',
        name='crazyflie_mapper_node',
        output='screen',
    )

    ekf_slam = Node(
        package='ekf_slam',
        executable='ekf_slam_node',
        name='ekf_slam_node',
        output='screen',
    )

    planning = Node(
        package='crazyflie_planning',
        executable='planning_node',
        name='planning_node',
        output='screen',
    )

    navigation = Node(
        package='crazyflie_navigation',
        executable='navigation_node',
        name='autonomous_navigation_node',
        output='screen',
    )

    box_landing = Node(
        package='crazyflie_navigation',
        executable='box_landing_node',
        name='box_landing_node',
        output='screen',
    )

    return LaunchDescription([
        crazyflie_hw,
        TimerAction(period=1.0, actions=[mapper]),
        TimerAction(period=2.0, actions=[ekf_slam]),
        TimerAction(period=3.0, actions=[planning]),
        TimerAction(period=4.0, actions=[navigation]),
        TimerAction(period=5.0, actions=[box_landing]),
    ])
