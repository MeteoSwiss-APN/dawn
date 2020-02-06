#TODO maybe function

if(NOT TARGET gtest)
  if(NOT FETCHED_GTEST)
    message(STATUS "Fetching GoogleTest...")
  endif()
  set(INSTALL_GTEST ON)
  set(BUILD_GMOCK OFF)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/release-1.10.0.tar.gz
  )
  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
  endif()
  set(FETCHED_GTEST ON CACHE BOOL "Fetched GoogleTest. Used to clean CMake output")
  mark_as_advanced(FETCHED_GTEST)
endif()
