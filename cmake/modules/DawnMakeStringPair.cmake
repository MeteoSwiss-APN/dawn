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
# dawn_make_string_pair
# ---------------------
#
# Create a formatted string of the given string pair.
#
#
# .. code-block:: cmake
#
#   dawn_make_string_pair(FIRST SECOND NUM_CHARS OUT_STRING)
#
# ``FIRST``
#   First string.
# ``SECOND``
#   Second string.
# ``NUM_CHARS``
#   Characters before ``SECOND`` is printed.
# ``OUT_STRING``
#   Output string.
#
# This is equivalent to the python snippet:
#
# .. code-block:: python
#
#   OUT_STRING = "%s%s%s" % (FIRST, " " * (NUM_CHARS - len(FIRST)), SECOND)
#
# Example
# ^^^^^^^
# .. code-block:: cmake
#
#   dawn_make_string_pair(Foo Bar1 10 out_var1)
#   dawn_make_string_pair(LongerFoo Bar2 10 out_var2)
#   message(${out_var1}) # "Foo       Bar1"
#   message(${out_var2}) # "LongerFoo Bar2"
#
function(dawn_make_string_pair FIRST SECOND NUM_CHARS OUT_STRING)
  set(max_length ${NUM_CHARS})
  string(LENGTH ${FIRST} first_name_length)
  math(EXPR indent_length "${max_length} - ${first_name_length}")

  set(full_indent "")
  foreach(var RANGE 0 ${NUM_CHARS})
    set(full_indent "${full_indent} ")
  endforeach()
  string(SUBSTRING ${full_indent} "0" "${indent_length}" indent)
  set(${OUT_STRING} "${FIRST}${indent}${SECOND}" PARENT_SCOPE)
endfunction()