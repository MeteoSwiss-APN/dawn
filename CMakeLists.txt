cmake_minimum_required(VERSION 3.13)
project(dawn-gtclang-multiproject)

include(CTest)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Include sub-projects
message(STATUS "Configuring Dawn...")
option(DAWN_BUILD_TESTING "Enable Dawn tests." ${BUILD_TESTING})
option(DAWN_BUILD_EXAMPLES "Enable Dawn examples." ${BUILD_TESTING})
add_subdirectory(dawn)

message(STATUS "Configuring GTClang...")
option(GTCLANG_BUILD_TESTING "Enable GTClang tests." ${BUILD_TESTING})
add_subdirectory(gtclang)
