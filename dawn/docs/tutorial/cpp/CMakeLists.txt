cmake_minimum_required(VERSION 2.8.9)

project(dawn_standalone)

include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../dawn/bundle/install/include/)
include_directories(${CMAKE_CURRENT_LIST_DIR}/../../../dawn/bundle/install/include/dawn/Support/External/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/../../../dawn/bundle/install/lib/)

set (CMAKE_CXX_STANDARD 17)

add_executable(dawn_standalone introDawnStandalone.cpp)
target_link_libraries(dawn_standalone DawnC)
