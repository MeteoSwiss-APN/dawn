cmake_minimum_required(VERSION 3.13)
project(dawn-gtclang-multiproject)

# Builds both dawn and gtclang
set(DAWN_GTCLANG_MULTIPROJECT ON CACHE BOOL "Build dawn and gtclang together")
mark_as_advanced(DAWN_GTCLANG_MULTIPROJECT)

# Include sub-projects
add_subdirectory(dawn)
add_subdirectory(gtclang)
