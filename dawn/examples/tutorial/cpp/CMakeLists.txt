cmake_minimum_required(VERSION 3.12.4)

project(dawn_standalone LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
add_executable(dawn_standalone introDawnStandalone.cpp)

set(DAWN_INCLUDE_DIR "" CACHE PATH "dawn include directory")
set(DAWN_EXTERNAL_INCLUDE_DIR "" CACHE PATH "dawn external include directory")
set(DAWN_LIBRARY_DIR "" CACHE PATH "dawn library directory")

find_path(DAWN_INCLUDE_DIRS
    NAMES dawn/Dawn.h 
    HINTS ${CMAKE_CURRENT_LIST_DIR}/../../../bundle/install/include/ ${DAWN_INCLUDE_DIR}
    )

find_path(DAWN_EXTERNAL_INCLUDE_DIRS
    NAMES nlohmann/json.hpp
    HINTS ${CMAKE_CURRENT_LIST_DIR}/../../../bundle/install/include/dawn/Support/External/ ${DAWN_EXTERNAL_INCLUDE_DIR}
    )

find_path(DAWN_LIBRARY_DIRS
    NAMES libDawn.so
    HINTS ${CMAKE_CURRENT_LIST_DIR}/../../../bundle/install/lib/ ${DAWN_LIBRARY_DIR}
    )

mark_as_advanced(DAWN_INCLUDE_DIRS DAWN_EXTERNAL_INCLUDE_DIRS DAWN_LIBRARY_DIRS)

#includes: dawn and dawn external (for reading the sir bytestream)
target_include_directories(dawn_standalone PRIVATE ${DAWN_INCLUDE_DIRS} ${DAWN_EXTERNAL_INCLUDE_DIRS})
target_link_directories(dawn_standalone PRIVATE ${DAWN_LIBRARY_DIRS})

message(STATUS ${DAWN_INCLUDE_DIRS})

#link against dawn-c interface
target_link_libraries(dawn_standalone DawnC)