# TODO maybe function

if(NOT DAWN_BUILT_PROTOBUF)
  message(STATUS " Fetching Protobuf...")
endif()

# Configuration options for Protobuf
set(protobuf_BUILD_EXAMPLES OFF)
set(protobuf_BUILD_TESTS OFF)
set(protobuf_INSTALL_EXAMPLES OFF)
set(protobuf_BUILD_PROTOC_BINARIES ON)

FetchContent_Declare(protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v3.10.1
  )

FetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
  FetchContent_Populate(protobuf)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
endif()

set(USE_BUNDLED_PROTOBUF ON CACHE BOOL "Use Protobuf from bundle.")
mark_as_advanced(USE_BUNDLED_PROTOBUF)

if(Python3_FOUND)
  add_custom_target(build-protobuf-python
    COMMAND PROTOC=$<TARGET_FILE:protobuf::protoc> ${Python3_EXECUTABLE} setup.py build
    WORKING_DIRECTORY ${protobuf_SOURCE_DIR}/python
    )
  add_custom_target(protobuf-python
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${protobuf_SOURCE_DIR}/python ${protobuf_BINARY_DIR}/python
    DEPENDS build-protobuf-python
    )
endif()
