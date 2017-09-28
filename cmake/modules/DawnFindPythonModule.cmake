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

include(DawnIncludeGuard)
dawn_include_guard()

#.rst:
# dawn_find_python_module
# -----------------------
#
# Try to find a specific python module.
#
# .. code-block:: cmake
#
#   dawn_find_python_module(MODULE [REQUIRED])
#
# ``MODULE``
#   Python module to find.
# ``REQUIRED``
#   If set, abort with an error if ``MODULE`` was not found.
#
# The module makes use of the ``PYTHON_EXECUTABLE`` which needs to be set before calling this 
# function (see 
# `FindPythonInterp <https://cmake.org/cmake/help/latest/module/FindPythonInterp.html>`_).
#
# Variables defined:
#
# ``PYTHON_<module>_FOUND``   
#   Python module is avialable where ``<module>`` is the python module to search for in all 
#   uppercase.
# ``PYTHON_<module>_PATH``    
#   Path to the module where ``<module>`` is the python module to search for in all uppercase.
#
function(dawn_find_python_module MODULE)
  string(TOUPPER ${MODULE} module_upper)

  if(NOT PYTHON_${module_upper}_FOUND)
    # Check if module is required
    if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
      set(${MODULE}_FIND_REQUIRED TRUE)
    endif()
  
    if(NOT DEFINED PYTHON_EXECUTABLE)
       if(${MODULE}_FIND_REQUIRED)
          message(FATAL_ERROR "Could NOT find ${MODULE}: PYTHON_EXECUTABLE not defined")
       else()
          message(STATUS "Could NOT find ${MODULE}: PYTHON_EXECUTABLE not defined")
       endif()
       return()
    endif()

    # A module's location is usually a directory, but for binary modules it's a .so file.
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" "-c" 
      "import re, ${MODULE}; print(re.compile('/__init__.py.*').sub('',${MODULE}.__file__))"
      RESULT_VARIABLE _${MODULE}_status 
      OUTPUT_VARIABLE _${MODULE}_location
      ERROR_QUIET 
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Set result
    if(NOT _${MODULE}_status)
      set(PYTHON_${module_upper}_PATH ${_${MODULE}_location} 
          CACHE STRING "Location of Python module ${MODULE}")
      set(PYTHON_${module_upper}_FOUND TRUE
          CACHE STRING "Python module ${MODULE} found")
      mark_as_advanced(PYTHON_${module_upper}_PATH PYTHON_${module_upper}_FOUND)
      message(STATUS "Found Python module ${MODULE}: ${PYTHON_${module_upper}_PATH}")
    else()
      set(PYTHON_${module_upper}_FOUND FALSE)
        if(${MODULE}_FIND_REQUIRED)
          message(FATAL_ERROR "Could NOT find module ${MODULE} for python ${PYTHON_EXECUTABLE}")
        else()
          message(STATUS "Could NOT find ${MODULE} for python: ${PYTHON_EXECUTABLE}")
        endif()
    endif()
  endif(NOT PYTHON_${module_upper}_FOUND)
endfunction(dawn_find_python_module)