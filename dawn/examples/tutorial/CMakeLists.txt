cmake_minimum_required(VERSION 3.3)
project(laplacian_driver LANGUAGES CXX)

find_package(Dawn REQUIRED)
find_package(GTClang REQUIRED)
add_executable(laplacian_driver laplacian_driver.cpp)
target_link_libraries(laplacian_driver PUBLIC GTClang::dsl_defs)
# defines symbol BACKEND_MC
target_compile_definitions(laplacian_driver PRIVATE BACKEND_MC)



# set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

# find_path(DAWN_INCLUDE_DIR
#     NAMES driver-includes/gridtools_includes.hpp
#     HINTS ${ROOT_DIR}/dawn/bundle/install/include ${DAWN_INCLUDE_DIR}
#     )

# find_path(GTCLANG_INCLUDE_DIR
#     NAMES gtclang_dsl_defs/gtclang_dsl.hpp
#     HINTS ${ROOT_DIR}/gtclang/bundle/install/include ${GTCLANG_INCLUDE_DIR}
#     )

# find_path(GRIDTOOLS_INCLUDE_DIR
#     NAMES gridtools/stencil_composition/stencil_composition.hpp
#     HINTS ${ROOT_DIR}/gtclang/bundle/build/gridtools-prefix/src/gridtools/include ${GRIDTOOLS_INCLUDE_DIR}
#     )

# mark_as_advanced(GRIDTOOLS_INCLUDE_DIR GTCLANG_INCLUDE_DIR DAWN_INCLUDE_DIR)

# # includes: dawn, gtclang and gridtools for GTClang example
# target_include_directories(laplacian_driver PRIVATE ${GTCLANG_INCLUDE_DIR} ${GRIDTOOLS_INCLUDE_DIR} ${DAWN_INCLUDE_DIR})
