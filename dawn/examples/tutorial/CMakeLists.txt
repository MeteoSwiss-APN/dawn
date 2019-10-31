cmake_minimum_required(VERSION 3.3)

project(laplacian_driver LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(laplacian_driver laplacian_driver.cpp)

# defines symbol BACKEND_MC
target_compile_definitions(laplacian_driver PRIVATE BACKEND_MC)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

find_path(GTCLANG_INCLUDE_DIR
    NAMES gridtools/clang_dsl.hpp
    HINTS ${ROOT_DIR}/gtclang/bundle/install/include ${GTCLANG_INCLUDE_DIR}
    )

find_path(GRIDTOOLS_INCLUDE_DIR
    NAMES gridtools/stencil_composition/stencil_composition.hpp
    HINTS ${ROOT_DIR}/gtclang/bundle/build/gridtools-prefix/src/gridtools/include ${GRIDTOOLS_INCLUDE_DIR}
    )

mark_as_advanced(GRIDTOOLS_INCLUDE_DIR GTCLANG_INCLUDE_DIR)

# includes: gtclang and gridtools for GTClang example
target_include_directories(laplacian_driver PRIVATE ${GTCLANG_INCLUDE_DIR} ${GRIDTOOLS_INCLUDE_DIR})
