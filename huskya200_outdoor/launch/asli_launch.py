from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

from launch.substitutions import PythonExpression

def generate_launch_description():
    # Declare launch args
    declare_urdf_file = DeclareLaunchArgument(
        'urdf_file',
        default_value='urdf/husky.urdf',
        description='Relative path to URDF file inside the package'
    )

    declare_pkg = DeclareLaunchArgument(
        'urdf_package',
        default_value='husky_description',
        description='Package where the URDF is located'
    )

    # Substitutions
    urdf_file = LaunchConfiguration('urdf_file')
    urdf_package = LaunchConfiguration('urdf_package')

    # Read URDF file directly
    robot_description_content = ParameterValue(
    PythonExpression([
        "open('", 
        PathJoinSubstitution([FindPackageShare(urdf_package), urdf_file]),
        "').read()"
    ]),
    value_type=str
    )
    robot_description = {'robot_description': robot_description_content}


    # … declare args, robot_description, rsp_node …

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': True}],
        output='screen'
    )

    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r',
             PathJoinSubstitution([
               FindPackageShare('huskya200_outdoor'),
               'worlds', 'husky_outdoor.sdf'
             ])],
        output='screen'
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name',  'husky',
            '-x', '33.13', '-y', '-4.78', '-z', '0.5'
        ],
        output='screen'
    )
    teleop_husky = Node(
    package='teleop_twist_keyboard',
    executable='teleop_twist_keyboard',
    remappings=[('/cmd_vel', '/model/husky/cmd_vel')],
    emulate_tty=True, 
    output='screen'
)


    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    
    arguments=[
       ' /world/empty/model/husky/link/base_link/sensor/camera_realsense_link/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
        '/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/image@sensor_msgs/msg/Image@gz.msgs.Image',
        '/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/points@sensor_msgs/msg/Image@gz.msgs.Image',
        '/gps@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat',
        '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
        '/lidar@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
        '/lidar/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
        '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
        '/model/husky/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry',
        '/clock@rosgraph_msgs/msg/Clock@gz.msgs.Clock',
        
    ],
    remappings=[
        ('/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/camera_info','/camera_info'),
        ('/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/depth_image','/image_depth'),
        ('/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/image','/image'),
        ('/world/empty/model/husky/link/base_link/sensor/camera_realsense_link/points','/image_ponits'),
        ('/gps','/gps'),
        ('/imu','/imu'),
        ('/lidar','/scan'),
        ('/lidar/points','/lidar/points'),
        ('/odom','/odom'),
        ('/cmd_vel','/cmd_vel'),
        ('/model/husky/odometry','/ground_truth'),
        ('/clock','/clock'),

      
    ],
    parameters=[{'use_sim_time': True}],
    )
    


    return LaunchDescription([
        declare_urdf_file,
        declare_pkg,
        #gz_sim,
        rsp_node,
        spawn_entity,
        bridge
       # teleop_husky
        
    ])
