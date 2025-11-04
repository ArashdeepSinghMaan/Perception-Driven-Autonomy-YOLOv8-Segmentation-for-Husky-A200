# localization_launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node

MAP_DIR = os.path.expanduser("~/maps/my_map")
MAP_YAML = os.path.join(MAP_DIR, "my_map.yaml")
AMCL_PARAMS = os.path.join(MAP_DIR, "amcl_params.yaml")

def generate_launch_description():
    return LaunchDescription([
        # 1) Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': MAP_YAML}]
        ),

        # 2) AMCL Localization
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[AMCL_PARAMS]
        ),
    ])
