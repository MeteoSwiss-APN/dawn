# TODO maybe function

if(NOT DAWN_BUILD_PROTOBUF)
  message(STATUS "Fetching Protobuf...")
endif()

# Configuration options for Protobuf
set(protobuf_BUILD_EXAMPLES OFF)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_INSTALL_EXAMPLES OFF)
set(protobuf_BUILD_PROTOC_BINARIES ON)

FetchContent_Declare(protobuf
  URL https://github.com/protocolbuffers/protobuf/archive/v3.10.1.tar.gz
)

FetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
  FetchContent_Populate(protobuf)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
endif()

set(DAWN_BUILD_PROTOBUF ON CACHE BOOL "Build Protobuf when building Dawn.")
mark_as_advanced(DAWN_BUILD_PROTOBUF)

if(Python3_FOUND)
  add_custom_command(OUTPUT ${protobuf_SOURCE_DIR}/python/build
    COMMAND PROTOC=$<TARGET_FILE:protobuf::protoc> ${Python3_EXECUTABLE} setup.py build
    WORKING_DIRECTORY ${protobuf_SOURCE_DIR}/python
    DEPENDS protobuf::protoc
  )
  add_custom_command(OUTPUT ${protobuf_BINARY_DIR}/python/google
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${protobuf_SOURCE_DIR}/python/build/lib ${protobuf_BINARY_DIR}/python
    DEPENDS ${protobuf_SOURCE_DIR}/python/build
  )
  add_custom_target(protobuf-python
    DEPENDS ${protobuf_BINARY_DIR}/python/google)
  if(DAWN_BUILD_PROTOBUF)
    install(DIRECTORY ${protobuf_SOURCE_DIR}/python/google DESTINATION ${CMAKE_INSTALL_LIBDIR}/python3)
  endif()
endif()
