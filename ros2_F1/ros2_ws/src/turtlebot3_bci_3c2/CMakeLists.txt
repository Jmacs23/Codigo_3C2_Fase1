cmake_minimum_required(VERSION 3.8)
project(turtlebot3_bci_3c2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- Mensajes propios ---
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TASMState.msg"
  "msg/BciCommand.msg"
  DEPENDENCIES std_msgs
)

# --- Modulo Python ---
ament_python_install_package(${PROJECT_NAME})

# --- Nodos ejecutables ---
install(PROGRAMS
  ${PROJECT_NAME}/bci_node.py
  ${PROJECT_NAME}/mission_node.py
  ${PROJECT_NAME}/ctrl_node.py
  ${PROJECT_NAME}/safety_node.py
  ${PROJECT_NAME}/interface_node.py
  ${PROJECT_NAME}/video_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# --- Launch y configuracion ---
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
