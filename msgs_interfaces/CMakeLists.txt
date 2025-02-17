cmake_minimum_required(VERSION 3.8)
project(msgs_interfaces)

# General settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GPT.srv"
  "srv/CameraParams.srv"
  "msg/MarkerPose.msg"
  "msg/MarkerPoseArray.msg"
  "msg/Point.msg"
  "msg/MarkerCorner.msg"
  "msg/SceneInfo.msg"
  DEPENDENCIES geometry_msgs sensor_msgs 
 )

ament_export_dependencies(rosidl_default_runtime)

ament_package()
