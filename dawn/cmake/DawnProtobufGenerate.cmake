
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

include(CMakeParseArguments)

#.rst:
# dawn_protobuf_generate
# ----------------------
#
# Run the protobuf compiler to generate sources from the proto files. This function excepts the
# Protobuf compiler to be imported as a target (``protobuf::protoc``).
#
# .. code-block:: cmake
#
#   dawn_protobuf_generate(OUT_FILES PROTOS LANGUAGE)
#
# ``OUT_FILES``
#   On output this variable contains a List of output files which contain the location of the header and
#   source files.
# ``OUTDIR``
#   Output directory. By default is set to CMAKE_CURRENT_BINARY_DIR.
# ``WDIR``
#   Working directory.
# ``PACKG``
#   Package directory (subdirectory of OUTDIR).
# ``PROTOS``
#   List of proto files to compile.
# ``LANGUAGE``
#   Language to compile to [default: cpp].
#
function(dawn_protobuf_generate)
  set(one_value_args OUT_FILES OUTDIR WDIR PACKG INC_DIR LANGUAGE)
  set(multi_value_args PROTOS)
  cmake_parse_arguments(ARG "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "dawn_protobuf_generate: invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  if(NOT ARG_PROTOS)
    message(FATAL_ERROR "dawn_protobuf_generate: called without any source files")
    return()
  endif()

  if(NOT ARG_WDIR)
    message(FATAL_ERROR "dawn_protobuf_generate: called without any workding directory argument")
    return()
  endif()
  if(NOT ARG_PACKG)
    message(FATAL_ERROR "dawn_protobuf_generate: called without any package argument")
    return()
  endif()

  if(NOT ARG_OUT_FILES)
    message(FATAL_ERROR
            "dawn_protobuf_generate: called without specifying the output variable (OUT_FILES)")
    return()
  endif()

  if(NOT ARG_LANGUAGE)
    set(ARG_LANGUAGE cpp)
  endif()

  if("${ARG_LANGUAGE}" STREQUAL "cpp")
    set(extensions .pb.h .pb.cc)
  elseif("${ARG_LANGUAGE}" STREQUAL "python")
    set(extensions _pb2.py)
  elseif("${ARG_LANGUAGE}" STREQUAL "java")
    set(extensions _pb2.java)
  else()
    message(FATAL_ERROR "dawn_protobuf_generate: unknown Language ${ARG_LANGUAGE}")
    return()
  endif()

  set(include_path "-I.")
  if(ARG_INC_DIR)
    set(include_path ${include_path} -I${ARG_INC_DIR})
  endif()

  # Set include directories based on protobuf target property
  get_property(Protobuf_INCLUDE_DIRS
    TARGET protobuf::libprotobuf
    PROPERTY INCLUDE_DIRECTORIES
  )
  foreach(incl ${Protobuf_INCLUDE_DIRS})
    set(include_path ${include_path} -I${incl})
  endforeach()

  # If protobuf is bundled, ensure it's built first
  if(DAWN_BUILD_PROTOBUF)
    set(depends protobuf-python)
  endif()

  set(output_files)

  if(ARG_OUTDIR)
    set(output_directory ${ARG_OUTDIR})
  else()
    set(output_directory ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  foreach(proto ${ARG_PROTOS})
    set(abs_file ${ARG_WDIR}/${proto})
    get_filename_component(basename ${proto} NAME_WE)

    unset(generated_srcs)
    foreach(ext ${extensions})
      if(${ARG_LANGUAGE} STREQUAL "java")
        list(APPEND generated_srcs "${output_directory}/dawn/sir/${basename}${ext}")
      else()
        list(APPEND generated_srcs "${output_directory}/${ARG_PACKG}/${basename}${ext}")
      endif()
    endforeach()

    add_custom_command(
      OUTPUT ${generated_srcs}
      COMMAND protobuf::protoc
      ARGS ${protobuf_script} --${ARG_LANGUAGE}_out ${output_directory}
           ${include_path} "${proto}"
      COMMENT "Running ${ARG_LANGUAGE} protocol buffer compiler on ${proto}"
      DEPENDS ${abs_file} protobuf::protoc ${depends}
      WORKING_DIRECTORY ${ARG_WDIR}
      VERBATIM
    )

    list(APPEND output_files ${generated_srcs})
  endforeach()

  set("${ARG_OUT_FILES}" "${output_files}" PARENT_SCOPE)
endfunction()
