from launch import LaunchDescription
from launch.actions import TimerAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Path to gz_sim.launch.py
    gz_launch_path = PathJoinSubstitution([
        FindPackageShare('ros_gz_sim'),
        'launch',
        'gz_sim.launch.py'
    ])

    # Start ros_gz_sim’s gz_sim.launch.py
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch_path),
        launch_arguments={
            'gz_args': PathJoinSubstitution([
                FindPackageShare('huskya200_outdoor'),
                'worlds',
                'husky_outdoor.sdf'
            ])
        }.items()
    )


    return LaunchDescription([gz_launch ])
