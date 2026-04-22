cmake_minimum_required(VERSION 3.8)
project(crazyflie_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

include_directories(include)

# Navigation node
add_executable(navigation_node
  src/navigation_node.cpp
)
ament_target_dependencies(navigation_node
  rclcpp
  geometry_msgs
  nav_msgs
  std_srvs
)

# Box landing node
add_executable(box_landing_node
  src/box_landing_node.cpp
)
ament_target_dependencies(box_landing_node
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
)

install(TARGETS
  navigation_node
  box_landing_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
