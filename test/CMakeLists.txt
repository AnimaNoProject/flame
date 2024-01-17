enable_testing()
set(gtest_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/gtest")
find_package(GTest CONFIG REQUIRED)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(TEST_SOURCES
        main.cpp
        stereo/epipolar_geometry_test.cc
        utils/image_utils_test.cc
        utils/load_tracker_test.cc
        utils/rasterization_test.cc
        utils/stats_tracker_test.cc
        utils/visualization_test.cc)

add_executable(${PROJECT_NAME}_test ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_test)