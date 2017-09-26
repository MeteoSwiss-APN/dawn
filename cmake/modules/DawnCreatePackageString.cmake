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

include(DawnMakeStringPair)

#.rst:
# .. _dawn_create_package_string:
#
# dawn_create_package_string
# --------------------------
#
# Create a package string including the location of the library/include directory and version 
# string. Note that the package has to be exported via :ref:`dawn_export_package`.
#
# .. code-block:: cmake
#
#   dawn_create_package_string(NAME PACKAGE_STRING [NOT_FOUND_STRING])
#
# ``NAME``
#   Name of the package.
# ``PACKAGE_STRING``
#   Returned package string.
# ``NOT_FOUND_STRING``
#   String to print of the package was not found, defaults to "Not FOUND" (optional).
#
# Example
# ^^^^^^^
# .. code-block:: cmake
#
#   dawn_export_package(Foo FOUND TRUE LIBRARIES "/usr/lib/libfoo.so" VERSION "1.2.3")
#   dawn_create_package_string(Foo FooPackageStr)
#   message(${FooPackageStr})
#
function(dawn_create_package_string NAME PACKAGE_STRING)
  set(not_found_string "NOT found")
  if(NOT("${ARGV2}" STREQUAL ""))
    set(not_found_string "${ARGV2}")
  endif()

  string(TOUPPER ${NAME} package)
  if(NOT(DEFINED DAWN_${package}_FOUND) OR NOT(${DAWN_${package}_FOUND}))
     set(info "${not_found_string}")
  else()
    if(DEFINED DAWN_${package}_INCLUDE_DIRS)
      list(GET  DAWN_${package}_INCLUDE_DIRS 0 inc)
      set(info "${inc}")
    elseif(DEFINED DAWN_${package}_EXECUTABLE)
      list(GET DAWN_${package}_EXECUTABLE 0 exe)
      set(info "${exe}")
    else()
      set(info "found")
    endif()

    if(DEFINED DAWN_${package}_VERSION)
      set(info "${info} (ver ${DAWN_${package}_VERSION})")
    endif()
  endif()

  dawn_make_string_pair(${NAME} ${info} 20 out_string)
  set(${PACKAGE_STRING} ${out_string} PARENT_SCOPE)
endfunction()
