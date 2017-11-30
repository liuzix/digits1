cmake_minimum_required(VERSION 3.5)
project(digits)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp network.cpp network.h)
add_executable(digits ${SOURCE_FILES})
target_link_libraries(digits ${ARMADILLO_LIBRARIES})