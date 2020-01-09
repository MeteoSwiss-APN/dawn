cmake_minimum_required(VERSION 3.13)
project(dawn_standalone LANGUAGES CXX)

find_package(Dawn REQUIRED)

add_executable(dawn_standalone dawn_standalone.cpp)
target_link_libraries(dawn_standalone PUBLIC Dawn::DawnCShared)
