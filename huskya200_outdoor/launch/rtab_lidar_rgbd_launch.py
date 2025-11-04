#!/usr/bin/env python3
"""
Minimal RTAB-Map launch for RGB-D + 2D LiDAR
Place in your_package/launch/rgbd_lidar_launch.py
Make sure you have lidar+rgbd params set in your_package/config/rgbd_lidar_params.yaml
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Path to params YAML
    config_file = os.path.join(
        get_package_share_directory('warehouse_robot'),
        'config',
        'rgbd_lidar_params.yaml'
    )

    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='true',
            description='Use simulation (Gazebo) clock'
        ),
        DeclareLaunchArgument(
            'scan_topic', default_value='/scan',
            description='Input topic for 2D LiDAR scans'
        ),
        DeclareLaunchArgument(
            'rgb_topic', default_value='/image',
            description='RGB image topic'
        ),
        DeclareLaunchArgument(
            'depth_topic', default_value='/image_depth',
            description='Depth image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='/camera_info',
            description='Camera info topic'
        ),
        DeclareLaunchArgument(
            'map_frame_id', default_value='map',
            description='Output map frame id'
        ),
        DeclareLaunchArgument(
            'odom_topic', default_value='/odometry/filtered',  description='Odometry topic name.'),

        # RTAB-Map SLAM Node (RGB-D + LiDAR)
        Node(
            package='rtabmap_slam',
            executable='rtabmap',
            name='rtabmap_rgbd_lidar',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'frame_id': 'base_link',
                    'map_frame_id': LaunchConfiguration('map_frame_id'),
                    'subscribe_depth': True,
                    'subscribe_rgb': True,
                    'subscribe_scan': True
                }
            ],
            remappings=[
                ('scan', LaunchConfiguration('scan_topic')),
                ('rgb/image', LaunchConfiguration('rgb_topic')),
                ('depth/image', LaunchConfiguration('depth_topic')),
                ('rgb/camera_info', LaunchConfiguration('camera_info_topic')),
                ('odom', LaunchConfiguration('odom_topic')),
                ('map', '/map')

            ],
        ),

        
        
    ])
