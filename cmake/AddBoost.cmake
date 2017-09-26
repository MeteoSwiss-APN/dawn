##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _       
##                        | |     | |
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

set(Boost_USE_MULTITHREADED ON)
if(BUILD_SHARED_LIBS)
  set(Boost_USE_STATIC_LIBS OFF)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
  add_definitions(-DBOOST_ALL_DYN_LINK)
else()
  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME OFF)
endif()

set(GTCLANG_BOOST_COMPONENTS system)
find_package(Boost 1.58 COMPONENTS ${GTCLANG_BOOST_COMPONENTS} REQUIRED)

dawn_export_package(
  NAME Boost
  FOUND ${Boost_FOUND} 
  VERSION "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}" 
  LIBRARIES ${Boost_LIBRARIES}
  INCLUDE_DIRS ${Boost_INCLUDE_DIRS}
  DEFINITIONS -DBOOST_ALL_NO_LIB
)
