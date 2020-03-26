function(add_update_codegen_reference_target)
    set(options)
    set(oneValueArgs NAME STENCIL BACKEND)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(inputDir ${CMAKE_CURRENT_SOURCE_DIR}/../input)
    set(refsDir ${CMAKE_CURRENT_SOURCE_DIR}/../reference)
    set(refsBuildDir ${CMAKE_CURRENT_BINARY_DIR}/../reference)

    if(${ARG_BACKEND} STREQUAL cuda)
        set(extension cu)
    else()
        set(extension cpp)
    endif()

    add_custom_target(${ARG_NAME} 
        COMMAND dawn-codegen ${inputDir}/${ARG_STENCIL}.iir -o ${refsDir}/${ARG_STENCIL}.${extension} --backend=${ARG_BACKEND}
        COMMAND ${CMAKE_COMMAND} -E copy ${refsDir}/${ARG_STENCIL}.${extension} ${refsBuildDir}/${ARG_STENCIL}.${extension}
        DEPENDS dawn-codegen)
endfunction()

# Structured stencils
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceConditionalCuda STENCIL conditional_stencil BACKEND cuda)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceConditionalNaive STENCIL conditional_stencil BACKEND naive)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceCopyCuda STENCIL copy_stencil BACKEND cuda)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceCopyNaive STENCIL copy_stencil BACKEND naive)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceGlobalIndexCuda STENCIL global_index_stencil BACKEND cuda)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceGlobalIndexNaive STENCIL global_index_stencil BACKEND naive)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceHoriDiffCuda STENCIL hori_diff_stencil BACKEND cuda)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceHoriDiffNaive STENCIL hori_diff_stencil BACKEND naive)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceTridiagonalSolveCuda STENCIL tridiagonal_solve_stencil BACKEND cuda)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceTridiagonalSolveNaive STENCIL tridiagonal_solve_stencil BACKEND naive)

# Unstructured stencils
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceICONLaplacian STENCIL ICON_laplacian_stencil BACKEND cxx-naive-ico)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceSimpleReduction STENCIL simple_reduction_stencil BACKEND cxx-naive-ico)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceSparseDimensions STENCIL sparse_dimensions BACKEND cxx-naive-ico)
add_update_codegen_reference_target(NAME UpdateCodeGenReferenceTridiagonalSolveUnstructured STENCIL tridiagonal_solve_unstructured BACKEND cxx-naive-ico)

add_custom_target(UpdateCodeGenReferences)
add_dependencies(UpdateCodeGenReferences 
                    UpdateCodeGenReferenceConditionalCuda
                    UpdateCodeGenReferenceConditionalNaive
                    UpdateCodeGenReferenceCopyCuda
                    UpdateCodeGenReferenceCopyNaive
                    UpdateCodeGenReferenceGlobalIndexCuda
                    UpdateCodeGenReferenceGlobalIndexNaive
                    UpdateCodeGenReferenceHoriDiffCuda
                    UpdateCodeGenReferenceHoriDiffNaive
                    UpdateCodeGenReferenceTridiagonalSolveCuda
                    UpdateCodeGenReferenceTridiagonalSolveNaive
                    UpdateCodeGenReferenceICONLaplacian
                    UpdateCodeGenReferenceSimpleReduction
                    UpdateCodeGenReferenceSparseDimensions
                    UpdateCodeGenReferenceTridiagonalSolveUnstructured
                )