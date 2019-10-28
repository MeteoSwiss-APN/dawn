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

# Include the GTClang specific options, definitions and macros
include(GTClangOptions)
include(GTClangDefinitions)
include(GTClangMacros)

if(GTCLANG_HAS_CUDA)
  include(GTClangSetupCUDA)
  gtclang_setup_CUDA()
  find_package(SLURM)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set C++ flags (Note that the LLVM/Clang package might add some other flags)
gtclang_set_cxx_flags()

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  yoda_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("gtclang version" ${GTCLANG_FULL_VERSION})
make_config_string("Platform" ${YODA_PLATFORM_STRING})
make_config_string("Architecture" ${YODA_ARCHITECTURE_STRING})
make_config_string("Compiler" ${YODA_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${GTCLANG_ASSERTS})
yoda_report_result("Configuration summary" ${config_info})

# Include the packages (and set the correct, libraries, includes etc.)
foreach(package bash ccache Python3 clang-format Clang Threads)
  include(Add${package})
  yoda_create_package_string(${package} info)
  list(APPEND package_info ${info})

  string(TOUPPER ${package} PACKAGE)
  if(${PACKAGE}_FOUND)
    list(APPEND GTCLANG_EXTERNAL_LIBRARIES ${YODA_${PACKAGE}_LIBRARIES})
    list(APPEND GTCLANG_EXTERNAL_INCLUDE_DIRS ${YODA_${PACKAGE}_INCLUDE_DIRS})
    list(APPEND GTCLANG_EXTERNAL_DEFINITIONS ${YODA_${PACKAGE}_DEFINITIONS})
  endif()
endforeach()

if(GTCLANG_BUILD_GT_CPU_EXAMPLES OR GTCLANG_BUILD_GT_GPU_EXAMPLES OR GTCLANG_BUILD_CUDA_EXAMPLES)
  include(AddGridTools)
  if(GridTools_FOUND)
    yoda_create_package_string(GridTools info)
    list(APPEND package_info ${info})

    string(TOUPPER GridTools PACKAGE)
    list(APPEND GTCLANG_EXTERNAL_LIBRARIES ${YODA_${PACKAGE}_LIBRARIES})
    list(APPEND GTCLANG_EXTERNAL_INCLUDE_DIRS ${YODA_${PACKAGE}_INCLUDE_DIRS})
    list(APPEND GTCLANG_EXTERNAL_DEFINITIONS ${YODA_${PACKAGE}_DEFINITIONS})
  endif()
endif()

include_directories(SYSTEM ${GTCLANG_EXTERNAL_INCLUDE_DIRS})
add_definitions(${GTCLANG_EXTERNAL_DEFINITIONS})

# Output summary of the packages
yoda_report_result("Package summary" ${package_info})

# TODO: Determine best method to get Protobuf_LIBRARIES variable here and add to rpath

# Build gtclang
include_directories(${DAWN_SOURCE_DIR}/src)
include_directories(${DAWN_BINARY_DIR}/src)

include_directories(${GTCLANG_SOURCE_DIR}/src)
include_directories(${GTCLANG_BINARY_DIR}/src)

add_subdirectory(src)

if(GTCLANG_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GTCLANG_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(GTCLANG_DOCUMENTATION)
  add_subdirectory(docs)
endif()

gtclang_gen_install_config()

# Install headers
install(
  DIRECTORY src/
  DESTINATION ${GTCLANG_INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)
