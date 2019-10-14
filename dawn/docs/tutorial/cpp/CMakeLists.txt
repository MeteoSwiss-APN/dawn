cmake_minimum_required(VERSION 2.8.9)

#includes: gtclang and gridtools for GTClang example
include_directories(~/workspace/dawn/dawn/bundle/install/include/)
include_directories(~/workspace/dawn/dawn/bundle/install/include/dawn/Support/External/)
link_directories(~/workspace/dawn/dawn/bundle/install/lib/)

project(dawn_standalone)

set (CMAKE_CXX_STANDARD 17)

add_executable(dawn_standalone introDawnStandalone.cpp)
target_link_libraries(dawn_standalone DawnC)
