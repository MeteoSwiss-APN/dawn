include(GoogleTest)

set(executable ${PROJECT_NAME}DriverIncludesUnittest)
add_executable(${executable}
  TestExtent.cpp
)

target_link_libraries(${executable} gtest gtest_main)
target_include_directories(${executable} PRIVATE ${PROJECT_SOURCE_DIR}/src)
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::DriverIncludes::Unit::" DISCOVERY_TIMEOUT 30)
