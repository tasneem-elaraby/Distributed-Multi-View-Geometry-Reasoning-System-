cmake_minimum_required(VERSION 3.8)
project(geometry_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeypointArray.msg"
  "msg/DescriptorArray.msg"
  "msg/MatchArray.msg"
  "msg/GeometricInliers.msg"
  "msg/CameraMotion.msg"
  "srv/CheckGeometry.srv"
  "action/ReportAction.action"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
