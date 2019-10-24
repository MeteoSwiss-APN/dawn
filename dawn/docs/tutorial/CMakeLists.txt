cmake_minimum_required(VERSION 3.12.4)

project(laplacian_driver LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(laplacian_driver laplacian_driver.cpp)

# defines symbol BACKEND_MC
target_compile_definitions(laplacian_driver PRIVATE BACKEND_MC)

find_path(gtclang_install 
    NAMES clang_dsl.hpp 
    #PATHS ${CMAKE_CURRENT_LIST_DIR}/../../../gtclang/bundle/install/include/gridtools
    PATHS /dev/null
    PATH_SUFFIXES include
    )

find_path(gridtools_install 
    NAMES stencil_functions.hpp 
    #PATHS ${CMAKE_CURRENT_LIST_DIR}/../../../gtclang/bundle/build/gridtools-prefix/src/gridtools/include/gridtools/stencil_composition
    PATHS /dev/null
    PATH_SUFFIXES gridtools/include
    )

if(gtclang_install)
    set(GTCLANG_INCLUDE_DIR "${gtclang_install}/../" CACHE PATH "gtclang include directory")
endif()

if(gridtools_install)
    set(GRIDTOOLS_INCLUDE_DIR "${gridtools_install}/../../" CACHE PATH "gridtools include directory")
endif()

message(STATUS ${GTCLANG_INCLUDE_DIR})
message(STATUS ${GRIDTOOLS_INCLUDE_DIR})

# includes: gtclang and gridtools for GTClang example
target_include_directories(laplacian_driver PRIVATE ${GTCLANG_INCLUDE_DIR} ${GRIDTOOLS_INCLUDE_DIR})

#example suffix ()