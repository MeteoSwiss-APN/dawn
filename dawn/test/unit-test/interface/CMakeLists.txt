##===------------------------------------------------------------------------------*- CMake -*-===##
##                          _
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

include(GoogleTest)

if(DAWN_REQUIRE_UNSTRUCTURED_TESTING)
  find_package(eckit REQUIRED)
  find_package(atlas REQUIRED)
else()
  find_package(eckit QUIET)
  if(NOT eckit_FOUND)
    message(STATUS "Could not locate eckit.")
  endif()
  find_package(atlas QUIET)
  if(NOT atlas_FOUND)
    message(STATUS "Could not locate atlas.")
  endif()
endif()

if(eckit_FOUND AND atlas_FOUND)

  set(AtlasExecutable ${PROJECT_NAME}AtlasInterfaceUnittest)
  add_executable(${AtlasExecutable}
    TestAtlasInterface.cpp
  )

  set(ToylibExecutable ${PROJECT_NAME}ToylibInterfaceUnittest)
  add_executable(${ToylibExecutable}
    TestToylibInterface.cpp
  )

  target_add_dawn_standard_props(${AtlasExecutable})
  target_add_dawn_standard_props(${ToylibExecutable})

  target_include_directories(${AtlasExecutable} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_include_directories(${ToylibExecutable} PRIVATE ${CMAKE_SOURCE_DIR}/src)

  target_link_libraries(${AtlasExecutable} atlas eckit gtest gtest_main)
  target_link_libraries(${ToylibExecutable} toylib gtest gtest_main)

  gtest_discover_tests(${AtlasExecutable} TEST_PREFIX "Dawn::Unit::Interface::" DISCOVERY_TIMEOUT 30)
  gtest_discover_tests(${ToylibExecutable} TEST_PREFIX "Dawn::Unit::Interface::" DISCOVERY_TIMEOUT 30)

else()
  message(STATUS "unstructured interface tests require both eckit and atlas. Disabling...")
endif()