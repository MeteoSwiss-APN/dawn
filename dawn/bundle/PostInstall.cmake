cmake_policy(SET CMP0057 NEW)
cmake_policy(SET CMP0011 NEW)
set(examples "copy_stencil;hori_diff;tridiagonal_solve;unstructured_stencil")
set(verified_examples "copy_stencil;hori_diff;tridiagonal_solve")

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(PythonInterp 3.5 REQUIRED)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/python/dawn/test_sir.py RESULT_VARIABLE res)

if(NOT ${res} EQUAL 0)
  message(FATAL_ERROR "SIR test failed")
else()
  message("SIR test succeded")
endif()

set(ENV{PYTHONPATH} "$ENV{PYTHONPATH}:${CMAKE_INSTALL_PREFIX}/python")
foreach(example ${examples})
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/examples/python/${example}.py RESULT_VARIABLE res)
  if(NOT ${res} EQUAL 0)
    message(FATAL_ERROR "SIR Python example did not run. TEST failed")
  endif()

  if(${example} IN_LIST verified_examples)
    execute_process(COMMAND diff ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}.cpp ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}_ref.cpp RESULT_VARIABLE res)

    if(NOT ${res} EQUAL 0)
      message(FATAL_ERROR "SIR Python example ${example} TEST failed")
    else() 
      message("SIR Python example ${example} TEST succeded")
    endif()
  endif()
endforeach()
