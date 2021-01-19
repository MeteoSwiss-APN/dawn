include(GoogleTest)

set(executable ${PROJECT_NAME}DriverIncludesUnittest)
add_executable(${executable}
  TestExtent.cpp
)

target_link_libraries(${executable} gtest gtest_main)
target_include_directories(${executable} PRIVATE ${PROJECT_SOURCE_DIR}/src)
# force to c++11 as generated code needs to be c++11 compliant
set_target_properties(${executable} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
gtest_discover_tests(${executable} TEST_PREFIX "Dawn::DriverIncludes::Unit::" DISCOVERY_TIMEOUT 30)
