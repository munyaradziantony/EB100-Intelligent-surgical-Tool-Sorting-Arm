import os
import launch
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    # Set the environment variable for Ignition Gazebo resources
    os.environ["IGN_GAZEBO_RESOURCE_PATH"] = os.path.join(
        FindPackageShare('surgicalBot').find("surgical_tools_sorter"), 'models')

    # Launch Ignition Gazebo with the specified world
    ign_gazebo_world = ExecuteProcess(
        cmd=[
            'ign', 'gazebo',  # Change this line
            PathJoinSubstitution([
                FindPackageShare('surgical_tools_sorter'),
                'worlds',
                'just_arm.sdf'
            ]),
            '--render-engine', 'ogre2',  # Set the rendering engine to Ogre
            '-v', '4'  # Set verbosity level to 4 for more detailed logs
        ],
        shell=True
    )

    # Launch the ROS 2 to Ignition bridge for the camera topic
    cam_topic_bridge = Node(
        package='ros_ign_bridge',
        executable='parameter_bridge',
        arguments=['/camera/image@sensor_msgs/msg/Image@ignition.msgs.Image'],
    )
    
    # Launch the ROS 2 to Ignition bridge for joint angles
    joint_angle_bridges = [
        Node(
            package='ros_ign_bridge',
            executable='parameter_bridge',
            arguments=[f'/joint_angles/joint{i}@std_msgs/msg/Float64@ignition.msgs.Double']  # Adjust message types as needed
        ) for i in range(1, 7)
    ] + [
        Node(
            package='ros_ign_bridge',
            executable='parameter_bridge',
            arguments=[f'/joint_angles/gripper_{side}_joint@std_msgs/msg/Float64@ignition.msgs.Double']
        ) for side in ['left', 'right']
    ]

    # Return the launch description with all actions
    return launch.LaunchDescription([
        ign_gazebo_world,
        cam_topic_bridge,
        *joint_angle_bridges,
    ])