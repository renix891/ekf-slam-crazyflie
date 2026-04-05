from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'gslam_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sixym3',
    maintainer_email='xiao.yuan0217@gmail.com',
    description='ROS 2 port of gslam Crazyflie controller',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = gslam_ros2.controller:main',
            'mapper_2d = gslam_ros2.mapper_2d_node:main',
            'planner_node = gslam_ros2.planner_node:main',
            'nav_node = gslam_ros2.nav_node:main',
            'mission_manager_node = gslam_ros2.mission_manager_node:main',
            'edge_detector_node = gslam_ros2.edge_detector_node:main',
            'range_monitor_node = gslam_ros2.range_monitor_node:main',
            'send_goal = gslam_ros2.send_goal:main',
        ],
    },
)
