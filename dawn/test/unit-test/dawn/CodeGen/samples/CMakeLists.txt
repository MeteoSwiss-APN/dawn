function(add_dawn4py_target)
    set(options)
    set(oneValueArgs NAME SCRIPT)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_custom_target(${ARG_NAME}
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${DAWN4PY_MODULE_DIR}:${PROTOBUF_PYTHON_DIR}
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../input
    )
endfunction()

add_dawn4py_target(NAME GenerateCodeGenInputCopy SCRIPT copy_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputGlobalIndex SCRIPT global_index_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputHoriDiff SCRIPT hori_diff_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputICONLaplacian SCRIPT ICON_laplacian_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputSimpleReduction SCRIPT simple_reduction_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputSparseDimensions SCRIPT sparse_dimensions.py)
add_dawn4py_target(NAME GenerateCodeGenInputTridiagonalSolve SCRIPT tridiagonal_solve_stencil.py)
add_dawn4py_target(NAME GenerateCodeGenInputTridiagonalSolveUnstructured SCRIPT tridiagonal_solve_unstructured.py)

add_custom_target(GenerateInputsCodeGenUnittests)
add_dependencies(GenerateInputsCodeGenUnittests 
                    GenerateCodeGenInputCopy
                    GenerateCodeGenInputGlobalIndex
                    GenerateCodeGenInputHoriDiff
                    GenerateCodeGenInputICONLaplacian
                    GenerateCodeGenInputSimpleReduction
                    GenerateCodeGenInputSparseDimensions
                    GenerateCodeGenInputTridiagonalSolve
                    GenerateCodeGenInputTridiagonalSolveUnstructured
                )