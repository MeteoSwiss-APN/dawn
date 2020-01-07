cmake_minimum_required(VERSION 3.3)
project(laplacian_driver LANGUAGES CXX)

find_package(Dawn REQUIRED)
find_package(GTClang REQUIRED)
add_executable(laplacian_driver laplacian_driver.cpp)
target_link_libraries(laplacian_driver PUBLIC GTClang::dsl_defs)
target_compile_definitions(laplacian_driver PRIVATE BACKEND_MC)
