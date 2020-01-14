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

set(EXEC_NAME ${PROJECT_NAME}UnittestCodeGenCuda)
add_executable(${EXEC_NAME} TestCodeGenCuda.cpp)
target_add_dawn_standard_props(${EXEC_NAME})
target_link_libraries(${EXEC_NAME} DawnUnittest gtest)
add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})
