function(add_dawn4py_target)
    set(options)
    set(oneValueArgs NAME SCRIPT)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_custom_target(${ARG_NAME}
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${DAWN4PY_MODULE_DIR}:${PROTOBUF_PYTHON_DIR}
        ${Python3_EXECUTABLE} ${ARG_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endfunction()

add_dawn4py_target(NAME GenerateSetLocationTypeInput SCRIPT SetLocationType.py)
