"""Bring up bag replay + TF broadcaster + RViz for a recording session.

The headline bags do not contain /tf or /tf_static, so RViz cannot resolve
the frames the messages live in. This launch fills that gap with the
small `bag_tf_broadcaster.py` node, then opens RViz with the saved
`replay_for_video.rviz` config.

Usage:
    ros2 launch ekf_slam replay_for_video.launch.py \\
        bag_path:=analysis/headline_bags/final_ekf_bag_run2

The bag plays at real-time with --clock so RViz timestamps match. Hit
the start button on your screen recorder, watch the flight, stop.

Default bag is the EKF Run 2 headline bag.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    project_dir = '/home/renix/EKF-SLAM-Autonomous-Crazyflie'
    default_bag = os.path.join(
        project_dir, 'analysis', 'headline_bags', 'final_ekf_bag_run2')

    pkg_dir = os.path.join(project_dir, 'ros2_workspace', 'src', 'ekf_slam')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'replay_for_video.rviz')
    broadcaster = os.path.join(pkg_dir, 'scripts', 'bag_tf_broadcaster.py')

    bag_arg = DeclareLaunchArgument(
        'bag_path', default_value=default_bag,
        description='Absolute path to the rosbag2 directory to replay.')

    # 1. TF broadcaster — runs from the source tree directly, no rebuild
    #    needed. use_sim_time so static transforms are stamped on bag clock.
    tf_node = ExecuteProcess(
        cmd=['python3', broadcaster,
             '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen',
    )

    # 2. RViz with the saved config. use_sim_time so display message-age
    #    filters compare to bag time.
    rviz_node = ExecuteProcess(
        cmd=['ros2', 'run', 'rviz2', 'rviz2',
             '-d', rviz_config,
             '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen',
    )

    # 3. Bag player. --clock publishes /clock so other nodes see bag time.
    #    Started last (after a small implicit delay from the prior two
    #    spawning) so the static transforms exist before consumers receive
    #    their first dynamic message.
    bag_player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play',
             LaunchConfiguration('bag_path'),
             '--clock'],
        output='screen',
    )

    return LaunchDescription([
        bag_arg,
        tf_node,
        rviz_node,
        bag_player,
    ])
