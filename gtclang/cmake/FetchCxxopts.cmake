if(NOT TARGET cxxopts::cxxopts)
  if(NOT INCLUDE_CXXOPTS)
    message(STATUS "Fetching cxxopts...")
  endif()
  set(CXXOPTS_BUILD_EXAMPLES OFF)
  set(CXXOPTS_BUILD_TESTS OFF)
  set(CXXOPTS_ENABLE_INSTALL ON)

  FetchContent_Declare(cxxopts
    URL https://github.com/jarro2783/cxxopts/archive/v2.2.0.tar.gz
  )

  FetchContent_GetProperties(cxxopts)
  if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR})
  endif()

  set(INCLUDE_CXXOPTS ON CACHE BOOL "Fetched cxxopts.")
  mark_as_advanced(INCLUDE_CXXOPTS)
endif()

add_library(cxxopts::cxxopts ALIAS cxxopts)
