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

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

function(add_gtclang_library)
  foreach(_l IN LISTS ARGN)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${_l})
    install(
      TARGETS ${_l}
      EXPORT ${PROJECT_NAME}Targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    add_library(${PROJECT_NAME}::${_l} ALIAS ${_l})
  endforeach()
endfunction()

add_subdirectory(Support)
add_gtclang_library(GTClangSupport)

add_subdirectory(Frontend)
add_gtclang_library(GTClangFrontend)

add_subdirectory(Driver)
add_gtclang_library(GTClangDriver)

if(${PROJECT_NAME}_TESTING)
  add_subdirectory(Unittest)
endif()

add_executable(gtclang Driver/gtclang.cpp)
set_target_properties(gtclang PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
install(
  TARGETS gtclang
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

target_link_libraries(gtclang GTClang)
