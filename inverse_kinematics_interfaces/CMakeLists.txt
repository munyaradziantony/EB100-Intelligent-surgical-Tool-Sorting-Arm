cmake_minimum_required(VERSION 3.8)
project(inverse_kinematics_interfaces)

find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  action/Posegoal.action
  msg/JointAngles.msg
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
