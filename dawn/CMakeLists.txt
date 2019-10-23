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

include(DawnCMakeInit)
set(DAWN_ROOT ${CMAKE_CURRENT_LIST_DIR})

dawn_cmake_init()

# Include the Dawn specific options, definitions and macros
include(DawnOptions)
include(DawnDefinitions)
include(DawnMacros)

# Set C++ standard
yoda_set_cxx_standard(c++17)

# Set C++ flags
dawn_set_cxx_flags()

# Add custom targets
yoda_add_target_clean_all(
  ${CMAKE_BINARY_DIR}/docs
  ${CMAKE_BINARY_DIR}/sphinx
  ${CMAKE_BINARY_DIR}/src
  ${CMAKE_BINARY_DIR}/test
  ${CMAKE_BINARY_DIR}/bin
  ${CMAKE_BINARY_DIR}/python
  ${CMAKE_BINARY_DIR}/dawn-cmake
)

# Output summary of the configuration
macro(make_config_string FIRST SECOND)
  yoda_make_string_pair(${FIRST} ": ${SECOND}" 20 out)
  list(APPEND config_info ${out})
endmacro()
make_config_string("Dawn version" ${DAWN_FULL_VERSION_STR})
make_config_string("Platform" ${YODA_PLATFORM_STRING})
make_config_string("Architecture" ${YODA_ARCHITECTURE_STRING})
make_config_string("Compiler" ${YODA_COMPILER_STRING})
make_config_string("Build type" ${CMAKE_BUILD_TYPE})
make_config_string("Asserts" ${DAWN_ASSERTS})
make_config_string("Build shared " ${BUILD_SHARED_LIBS})
make_config_string("Install prefix" ${CMAKE_INSTALL_PREFIX})
yoda_report_result("Configuration summary" ${config_info})

# Include the packages
if(DAWN_DOCUMENTATION)
  set(doc_packages Python3 Sphinx LaTeX Doxygen)
endif()

foreach(package bash Protobuf ccache clang-format ${doc_packages})
  include("Add${package}")
  yoda_create_package_string(${package} info)
  list(APPEND package_info ${info})

  string(TOUPPER ${package} PACKAGE)

  list(APPEND DAWN_EXTERNAL_LIBRARIES ${YODA_${PACKAGE}_LIBRARIES})
  list(APPEND DAWN_EXTERNAL_INCLUDE_DIRS ${YODA_${PACKAGE}_INCLUDE_DIRS})
  list(APPEND DAWN_EXTERNAL_DEFINITIONS ${YODA_${PACKAGE}_DEFINITIONS})

endforeach()

# Enable RPath support
yoda_enable_full_rpath("${CMAKE_INSTALL_PREFIX}/${DAWN_INSTALL_LIB_DIR};${DAWN_PROTOBUF_RPATH_DIR}")


include_directories(SYSTEM ${DAWN_EXTERNAL_INCLUDE_DIRS})
add_definitions(${DAWN_EXTERNAL_DEFINITIONS})

yoda_report_result("Package summary" ${package_info})

# Add clang-format target
yoda_add_target_clang_format(DIRECTORIES ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/test EXTENSION ".h;.cpp")

if(DAWN_PYTHON)
  if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(PythonInterp 3.4 REQUIRED)
  endif()
  add_subdirectory(python)
endif()

# Build dawn
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/test/utils/googletest/include)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/src/dawn/Support/External)

add_subdirectory(src)
add_subdirectory(cmake)

if(DAWN_EXAMPLES)
  add_subdirectory(examples)
endif()

if(DAWN_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()

if(DAWN_DOCUMENTATION)
  add_subdirectory(docs)
endif()

# Install headers
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/src/
  DESTINATION ${DAWN_INSTALL_INCLUDE_DIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc" PATTERN "*.hpp"
)

# Install CMake package configuration
dawn_gen_install_config()
