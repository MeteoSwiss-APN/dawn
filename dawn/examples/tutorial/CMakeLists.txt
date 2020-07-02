cmake_minimum_required(VERSION 3.13)
project(laplacian_driver LANGUAGES CXX)

find_package(Dawn REQUIRED)
find_package(GridTools 1.1.3 REQUIRED)

add_executable(laplacian_driver laplacian_driver.cpp)
target_compile_features(laplacian_driver PUBLIC cxx_std_14)
target_compile_definitions(laplacian_driver PRIVATE BACKEND_MC)
target_include_directories(laplacian_driver
  PUBLIC $<BUILD_INTERFACE:$<TARGET_PROPERTY:Dawn::Dawn,INTERFACE_INCLUDE_DIRECTORIES>>
         $<BUILD_INTERFACE:$<TARGET_PROPERTY:GridTools::gridtools,INTERFACE_INCLUDE_DIRECTORIES>>
         ${PROJECT_BINARY_DIR})
