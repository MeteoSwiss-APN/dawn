set(examples "copy_stencil;hori_diff;tridiagonal_solve")

if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(PythonInterp 3.5 REQUIRED)
endif()

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/python/dawn/test_sir.py RESULT_VARIABLE res)

if(NOT ${res} EQUAL 0)
  message(FATAL_ERROR "SIR test failed")
else()
  message("SIR test succeded")
endif()

set(ENV{PYTHONPATH} "ENV{PYTHONPATH}:${CMAKE_INSTALL_PREFIX}/python")
foreach(example ${examples})
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_INSTALL_PREFIX}/examples/python/${example}.py)
  execute_process(COMMAND diff ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}.cpp ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}_ref.cpp RESULT_VARIABLE res)

  if(NOT ${res} EQUAL 0) 
	  message(FATAL_ERROR "SIR Python example ${example} TEST failed")
  else() 
    message("SIR Python example ${example} TEST succeded")
  endif()
endforeach()
