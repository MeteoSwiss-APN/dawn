if(NOT DEFINED CCACHE_EXECUTABLE)
  find_program(CCACHE_EXECUTABLE
    NAMES ccache
    DOC "Path to ccache executable"
  )
  mark_as_advanced(CCACHE_EXECUTABLE)
endif()
if(CCACHE_EXECUTABLE)
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
endif()
