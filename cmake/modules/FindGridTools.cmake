if( NOT GRIDTOOLS_FOUND )

  find_path( GRIDTOOLS_INCLUDE_DIR 
             NAMES gridtools.hpp
             PATHS 
                ${CMAKE_INSTALL_PREFIX}
                "${GRIDTOOLS_PATH}"
                "${GRIDTOOLS_ROOT}"
                ENV GRIDTOOLS_PATH 
            PATH_SUFFIXES include
  )

  include(FindPackageHandleStandardArgs)

  # handle the QUIETLY and REQUIRED arguments and set GRIDTOOLS_FOUND to TRUE
  find_package_handle_standard_args( gridtools DEFAULT_MSG
                                     GRIDTOOLS_INCLUDE_DIR )

  mark_as_advanced( GRIDTOOLS_INCLUDE_DIRS GRIDTOOLS_LIBRARIES )

  set( gridtools_FOUND ${GRIDTOOLS_FOUND} )
  set( GRIDTOOLS_INCLUDE_DIRS 
    ${GRIDTOOLS_INCLUDE_DIR}
  )
endif()

