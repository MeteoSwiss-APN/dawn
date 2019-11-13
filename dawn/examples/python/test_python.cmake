cmake_minimum_required(VERSION 3.12.4)
if(NOT DEFINED PYTHON_EXECUTABLE)
    find_package(PythonInterp 3 REQUIRED)
endif()

if(NOT DEFINED DAWN_PYTHON_MODULES_DIR)
    message(FATAL_ERROR "DAWN_PYTHON_MODULES_DIR not set")
endif()
if(NOT DEFINED DAWN_PYTHON_EXAMPLES_DIR)
    message(FATAL_ERROR "DAWN_PYTHON_EXAMPLES_DIR not set")
endif()
if(NOT DEFINED EXAMPLE) # example to test
    message(FATAL_ERROR "EXAMPLE not set")
endif()
if(NOT DEFINED VERIFY)
    message(FATAL_ERROR "VERIFY not set")
endif()

set(ENV{PYTHONPATH} "$ENV{PYTHONPATH}:${DAWN_PYTHON_MODULES_DIR}") # TODO fix path

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${DAWN_PYTHON_EXAMPLES_DIR}/${EXAMPLE}.py RESULT_VARIABLE res) # TODO fix path
if(NOT ${res} EQUAL 0)
    message(FATAL_ERROR "SIR Python example ${EXAMPLE} did not run. TEST failed")
endif()

if(VERIFY)
    execute_process(COMMAND diff ${DAWN_PYTHON_EXAMPLES_DIR}/data/${EXAMPLE}.cpp ${DAWN_PYTHON_EXAMPLES_DIR}/data/${EXAMPLE}_ref.cpp RESULT_VARIABLE res)
endif()

if(NOT ${res} EQUAL 0)
    message(FATAL_ERROR "SIR Python example ${EXAMPLE} TEST failed")
else()
    message("SIR Python example ${EXAMPLE} TEST succeded")
endif()