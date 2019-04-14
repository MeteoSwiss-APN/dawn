set(examples "copy_stencil;hori_diff;tridiagonal_solve")

foreach(example ${examples})
  execute_process(COMMAND python3 ${CMAKE_INSTALL_PREFIX}/examples/python/${example}.py)
  execute_process(COMMAND diff ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}.cpp ${CMAKE_INSTALL_PREFIX}/examples/python/data/${example}_ref.cpp RESULT_VARIABLE res)

  if(NOT ${res} EQUAL 0) 
	  message(FATAL_ERROR "SIR Python example TESTS failed")
  else() 
    message("SIR Python example TESTS succeded")
  endif()
endforeach()
