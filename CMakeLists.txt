cmake_minimum_required(VERSION 3.27.8)

project(flame)
set(${PROJECT_NAME}_VERSION 1.0.0)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Use GTest to build and test libraries" ON)

# Locate external dependencies
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv3")
find_package(OpenCV)
set(BOOST_ROOT ${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET})
set(Boost_NO_SYSTEM_PATHS ON)
set(Boost_NO_BOOST_CMAKE  ON)
find_package(Boost REQUIRED)
set(Eigen3_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
set(fmt_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/fmt")
find_package(fmt CONFIG REQUIRED)
set(Sophus_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/Sophus")
find_package(Sophus CONFIG REQUIRED)

# Project includes
include_directories(SYSTEM src
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS})

# Build the library
add_subdirectory(src)

# If the BUILD_TESTS option is set, a Google test binary will be compiled.
if (BUILD_TESTS)
  add_subdirectory(./test)
endif()
