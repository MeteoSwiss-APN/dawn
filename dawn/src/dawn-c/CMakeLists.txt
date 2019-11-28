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

add_library(${PROJECT_NAME}CObjects OBJECT
  Dawn.h
  Compiler.cpp
  Compiler.h
  ErrorHandling.cpp
  ErrorHandling.h
  Options.cpp
  Options.h
  TranslationUnit.cpp
  TranslationUnit.h
  Types.h
  util/Allocate.h
  util/CompilerWrapper.h
  util/OptionsWrapper.cpp
  util/OptionsWrapper.h
  util/TranslationUnitWrapper.h
)
target_add_dawn_standard_props(${PROJECT_NAME}CObjects)
target_link_libraries(${PROJECT_NAME}CObjects PUBLIC ${PROJECT_NAME})

# build shared library unconditionally as needed for python
add_library(${PROJECT_NAME}CShared SHARED $<TARGET_OBJECTS:${PROJECT_NAME}CObjects>)
set_target_properties(${PROJECT_NAME}CShared PROPERTIES OUTPUT_NAME ${PROJECT_NAME}C)
target_link_libraries(${PROJECT_NAME}CShared PUBLIC ${PROJECT_NAME})

if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME}C ALIAS ${PROJECT_NAME}CShared)
else()
  add_library(${PROJECT_NAME}CStatic STATIC $<TARGET_OBJECTS:${PROJECT_NAME}CObjects>)
  set_target_properties(${PROJECT_NAME}CStatic PROPERTIES OUTPUT_NAME ${PROJECT_NAME}C)
  target_link_libraries(${PROJECT_NAME}CStatic PUBLIC ${PROJECT_NAME})
  add_library(${PROJECT_NAME}C ALIAS ${PROJECT_NAME}CStatic)
endif()
