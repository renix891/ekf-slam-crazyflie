cmake_minimum_required(VERSION 3.8)
project(crazyflie_planning)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# Include directories
include_directories(include)

# D* Lite library
add_library(dstarlite
  src/dstarlite.cpp
)

target_include_directories(dstarlite PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(dstarlite
  rclcpp
)

# Planning node executable
add_executable(planning_node
  src/planning_node.cpp
)

target_link_libraries(planning_node
  dstarlite
)

ament_target_dependencies(planning_node
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)

# Install targets
install(TARGETS
  dstarlite
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS
  planning_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY
  include/
  DESTINATION include
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
