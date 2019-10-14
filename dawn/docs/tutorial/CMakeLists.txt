cmake_minimum_required(VERSION 2.8.9)

#dfines symbol BACKEND_MC
add_definitions(-DBACKEND_MC)

#includes: gtclang and gridtools for GTClang example
include_directories(~/workspace/dawn/gtclang/bundle/install/include/)
include_directories(~/workspace/dawn/gtclang/bundle/build/gridtools-prefix/src/gridtools/include/)

project (laplacian_driver)

set (CMAKE_CXX_STANDARD 17)

add_executable(laplacian_driver laplacian_driver.cpp)
