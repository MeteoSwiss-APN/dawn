#TODO not working

# if(${PROJECT_NAME}_TESTING)
#   # configure dawn4py in the build directory for testing
#   file(COPY ${PROJECT_SOURCE_DIR}/src/dawn4py/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py)
#   add_custom_target(link_dawn4py_test ALL
#     COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:_dawn4py> ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dawn4py/$<TARGET_FILE_NAME:_dawn4py>
#   )

#   if(NOT DEFINED PYTHON_EXECUTABLE)
#     find_package(PythonInterp 3 REQUIRED)
#   endif()
#   add_test(
#     NAME dawn4py-test
#     COMMAND
#       ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}
#       ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}
#   )
# endif()
