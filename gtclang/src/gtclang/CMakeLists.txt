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

add_subdirectory(Support)
# include(yodaAddExecutable)
# include(yodaCombineLibraries)

# add_subdirectory(Support)
# add_subdirectory(Frontend)
# add_subdirectory(Driver)

# if(GTCLANG_TESTING)
#   add_subdirectory(Unittest)
# endif()

# # Combine the relevant object libraries into libGTClang.a
# yoda_combine_libraries(
#   NAME GTClang
#   OBJECTS GTClangDriverObjects
#           GTClangFrontendObjects
#           GTClangSupportObjects
#   INSTALL_DESTINATION ${GTCLANG_INSTALL_LIB_DIR}
#   VERSION ${GTCLANG_FULL_VERSION}
# )

# # Link the driver
# yoda_add_executable(
#   NAME gtclang
#   SOURCES Driver/gtclang.cpp
#   DEPENDS GTClangStatic
#           ${GTCLANG_EXTERNAL_LIBRARIES}
#   OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin
#   INSTALL_DESTINATION ${GTCLANG_INSTALL_BIN_DIR}
# )
