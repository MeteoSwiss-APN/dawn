macro( gtclang_add_nvcc_flags )
  foreach( flag ${ARGV} )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${flag} )
  endforeach()
endmacro()

macro( gtclang_setup_CUDA )

  find_package( CUDA 8.0 REQUIRED )
  message( STATUS "CUDA detected: " ${CUDA_VERSION} )

  set( CUDA_PROPAGATE_HOST_FLAGS ON )
  set( CUDA_LIBRARIES "" ) # Is this required?

  set( CUDA_ARCH "CUDA_ARCH-NOTFOUND" CACHE STRING "CUDA architecture (e.g. sm_35, sm_37, sm_60); precedence over GPU_DEVICE" )

  if( NOT CUDA_ARCH )

    if( ${GPU_DEVICE} STREQUAL "P100" )
      set( CUDA_ARCH "sm_60" )
    elseif( ${GPU_DEVICE} STREQUAL "K80"  )
      set( CUDA_ARCH "sm_37" )
    elseif( ${GPU_DEVICE} STREQUAL "K40"  )
      set( CUDA_ARCH "sm_35" )
    else()
      message( FATAL_ERROR "
        Could not deduce CUDA_ARCH from GPU_DEVICE=${GPU_DEVICE}.
        Possible options for GPU_DEVICE: K40, K80, P100.
            
        Alternatively:
          - Set CUDA_ARCH (e.g. for P100 : \"CUDA_ARCH=sm_60\")
          - Set ENABLE_CUDA=OFF
        " )
    endif()

  endif()

  list( APPEND GTCLANG_DEFINITIONS "-D_USE_GPU_" )

  string( REPLACE "." "" CUDA_VERSION_INT ${CUDA_VERSION} )
  gtclang_add_nvcc_flags( -DCUDA_VERSION=${CUDA_VERSION_INT} )
  gtclang_add_nvcc_flags( -arch=${CUDA_ARCH} )
  gtclang_add_nvcc_flags( --compiler-options -fPIC )

  if( ENABLE_WERROR )
      # Unfortunately we cannot treat all errors as warnings, we have to specify each warning.
      # The only supported warning in CUDA8 is cross-execution-space-call
    gtclang_add_nvcc_flags( --Werror cross-execution-space-call -Xptxas --warning-as-error --nvlink-options --warning-as-error )
  endif()

  # Suppress nvcc warnings
  foreach( _diag 
              dupl_calling_convention code_is_unreachable
              implicit_return_from_non_void_function
              calling_convention_not_allowed
              conflicting_calling_conventions )
    gtclang_add_nvcc_flags( -Xcudafe --diag_suppress=${_diag} )
  endforeach()
endmacro()

