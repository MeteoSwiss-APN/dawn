cmake_minimum_required(VERSION 3.3)

project(dawn_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(dawn_standalone dawn_standalone.cpp)

set(ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../..)

find_path(DAWN_INCLUDE_DIR
    NAMES dawn/Dawn.h 
    HINTS ${ROOT_DIR}/dawn/bundle/install/include ${DAWN_INCLUDE_DIR}
    )

find_path(DAWN_EXTERNAL_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS ${ROOT_DIR}/dawn/bundle/install/include/dawn/Support/External ${DAWN_EXTERNAL_INCLUDE_DIR}
    )

find_path(DAWN_LIBRARY_DIR
    NAMES libDawn.so
    HINTS ${ROOT_DIR}/dawn/bundle/install/lib/ ${DAWN_LIBRARY_DIR}
    )

mark_as_advanced(DAWN_INCLUDE_DIR DAWN_EXTERNAL_INCLUDE_DIR DAWN_LIBRARY_DIR)

# includes: dawn and dawn external (for reading the sir bytestream)
target_include_directories(dawn_standalone PRIVATE ${DAWN_INCLUDE_DIR} ${DAWN_EXTERNAL_INCLUDE_DIR})
target_link_directories(dawn_standalone PRIVATE ${DAWN_LIBRARY_DIR})

message(STATUS ${DAWN_INCLUDE_DIR})

# link against dawn-c interface
target_link_libraries(dawn_standalone DawnC)