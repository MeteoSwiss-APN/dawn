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

set(executable ${PROJECT_NAME}UnittestCodeGenCuda)
add_executable(${executable}
  TestCodeGenCuda.cpp
  TestMain.cpp
)
target_add_dawn_standard_props(${executable})
target_link_libraries(${executable} DawnUnittest gtest)
add_test(NAME ${executable} COMMAND ${executable})
file(COPY reference DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
