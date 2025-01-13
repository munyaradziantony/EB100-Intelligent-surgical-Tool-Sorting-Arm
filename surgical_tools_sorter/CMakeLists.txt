cmake_minimum_required(VERSION 3.8)
project(surgical_tools_sorter)

# Set compiler options for GNU C++ and Clang
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ros2_control REQUIRED)
find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(nav_msgs REQUIRED)

# Install directories
install(DIRECTORY
  launch
  models
  worlds
  DESTINATION share/${PROJECT_NAME}/
)

# Testing configuration
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Finalize the package
ament_package()