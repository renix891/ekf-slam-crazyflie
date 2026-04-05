cmake_minimum_required(VERSION 3.10)
project(hw3_mec559)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Problem 2 executable
add_executable(problem2 problem2.cpp)
target_link_libraries(problem2 Eigen3::Eigen)

# Problem 3 executable
add_executable(problem3 problem3.cpp)
target_link_libraries(problem3 Eigen3::Eigen)

# Problem 4 executable
add_executable(problem4 problem4.cpp)
target_link_libraries(problem4 Eigen3::Eigen)
