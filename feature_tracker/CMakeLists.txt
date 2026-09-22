cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_library(stereo_matcher src/stereo_matcher.cpp)
target_link_libraries(stereo_matcher ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )

target_link_libraries(feature_tracker stereo_matcher ${catkin_LIBRARIES} ${OpenCV_LIBS})

if(CATKIN_ENABLE_TESTING)
  catkin_add_gtest(stereo_matcher_test tests/stereo_matcher_test.cpp)
  if(TARGET stereo_matcher_test)
    target_link_libraries(stereo_matcher_test stereo_matcher ${catkin_LIBRARIES} ${OpenCV_LIBS})
  endif()
endif()
