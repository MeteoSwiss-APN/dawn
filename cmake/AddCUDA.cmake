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

find_package(CUDA REQUIRED 7.0)

set(CUDA_ARCH "sm_30" CACHE STRING "Compute capability for CUDA")
set(CUDA_ARCH_ENV $ENV{CUDA_ARCH})
if(CUDA_ARCH_ENV)
  set(CUDA_ARCH "${CUDA_ARCH_ENV}")
endif()

set(CUDA_NVCC_FLAGS 
  "${CUDA_NVCC_FLAGS}"
  "-D_USE_GPU_"
  "-arch=${CUDA_ARCH}" 
  "-Xcudafe" "--diag_suppress=dupl_calling_convention" 
  "-Xcudafe" "--diag_suppress=code_is_unreachable" 
  "-Xcudafe" "--diag_suppress=implicit_return_from_non_void_function" 
  "-Xcudafe" "--diag_suppress=calling_convention_not_allowed" 
  "-Xcudafe" "--diag_suppress=conflicting_calling_conventions"
  "-Xcudafe" "--diag_suppress=template_not_in_template"
)

# https://github.com/tensorflow/tensorflow/issues/1066
if(UNIX AND NOT APPLE)
  find_program(LSB_RELEASE lsb_release)
  if(LSB_RELEASE)
    execute_process(COMMAND ${LSB_RELEASE} -is
        OUTPUT_VARIABLE LSB_RELEASE_ID_SHORT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(${LSB_RELEASE_ID_SHORT} STREQUAL "Ubuntu")
      set(CXX_FLAGS "-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__")
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CXX_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
    endif()  
  endif()
endif()

# http://stackoverflow.com/questions/31940457/make-nvcc-output-traces-on-compile-error
if(${CUDA_VERSION_STRING} STREQUAL "7.0")
  add_definitions(-DBOOST_RESULT_OF_USE_TR1)
endif()

gtclang_export_package_variable(
  CUDA 
  ${CUDA_FOUND} 
  "CUDA: ${CUDA_VERSION_STRING} (compute capability ${CUDA_ARCH})" 
  ${CUDA_LIBRARIES}
)

