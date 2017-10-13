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

set(cwd ${CMAKE_CURRENT_LIST_DIR})

set(directories 
  ${cwd}/Accesses
  ${cwd}/Diagnostics
  ${cwd}/PassStageSplitter
  ${cwd}/PassMultiStageSplitter
  ${cwd}/PassSetCaches
  ${cwd}/PassStageReordering
  ${cwd}/PassStageMerger
  ${cwd}/PassFieldVersioning
  ${cwd}/PassTemporaryMerger
  ${cwd}/PassTemporaryType
  ${cwd}/PassPreprocessor
)

list(APPEND compiler_flags -std=c++11 -O2)
list(APPEND library_flags ${Boost_LIBRARIES})

list(APPEND gridtools_flags ${compiler_flags} ${include_flags} ${library_flags})
gtclang_add_integrationtest(NAME no-codegen
                            GTCLANG ${GTCLANG_EXECUTABLE}
                            DIRECTORIES ${directories}
                            GRIDTOOLS_FLAGS ${gridtools_flags}
)

if(GTCLANG_HAS_GRIDTOOLS)
  list(APPEND directories_with_codegen ${directories} ${cwd}/CodeGen)
  list(APPEND gridtools_flags_with_codegen ${gridtools_flags} -isystem${GRIDTOOLS_INCLUDE_DIRS} 
                                                              -isystem${Boost_INCLUDE_DIRS})
  foreach(include ${GTCLANG_DSL_INCLUDES})
    list(APPEND gridtools_flags_with_codegen -isystem${include})
  endforeach()

  gtclang_add_integrationtest(NAME all
                              GTCLANG ${GTCLANG_EXECUTABLE}
                              DIRECTORIES ${directories_with_codegen}
                              GRIDTOOLS_FLAGS ${gridtools_flags_with_codegen}
  )
endif()
