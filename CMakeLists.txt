cmake_minimum_required(VERSION 3.8)
set(NAME listener)
set(CMAKE_PREFIX_PATH ./libtorch) #./libtorch
set(CMAKE_CXX_COMPILER g++-11)
set(CMAKE_C_COMPILER gcc-11)
project(handshake)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(listener handshake.cpp)
ament_target_dependencies(listener rclcpp geometry_msgs)

install(TARGETS
  listener
  DESTINATION lib/${PROJECT_NAME})

target_link_libraries(${NAME} "${TORCH_LIBRARIES}")
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 17)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
