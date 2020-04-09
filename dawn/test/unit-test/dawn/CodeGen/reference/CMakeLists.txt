
add_custom_target(update_codegen_refs)
function(add_update_codegen_reference_target)
    set(options)
    set(oneValueArgs STENCIL)
    set(multiValueArgs BACKENDS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(input_dir ${CMAKE_CURRENT_SOURCE_DIR}/../input)
    set(ref_dir ${CMAKE_CURRENT_SOURCE_DIR}/../reference)
    set(ref_build_dir ${CMAKE_CURRENT_BINARY_DIR}/../reference)

    foreach(backend ${ARG_BACKENDS})
        if(${backend} STREQUAL cuda)
            set(extension cu)
        else()
            set(extension cpp)
        endif()

        set(target codegen_${ARG_STENCIL}_${backend})
        add_custom_target(${target}
            COMMAND dawn-codegen ${input_dir}/${ARG_STENCIL}.iir -o ${ref_dir}/${ARG_STENCIL}.${extension} --backend=${backend}
            COMMAND ${CMAKE_COMMAND} -E copy ${ref_dir}/${ARG_STENCIL}.${extension} ${ref_build_dir}/${ARG_STENCIL}.${extension}
            DEPENDS dawn-codegen)
        add_dependencies(update_codegen_refs ${target})
    endforeach()
endfunction()

# Structured stencils
add_update_codegen_reference_target(STENCIL conditional_stencil BACKENDS c++-naive cuda)
add_update_codegen_reference_target(STENCIL copy_stencil BACKENDS c++-naive cuda)
add_update_codegen_reference_target(STENCIL global_index_stencil BACKENDS c++-naive cuda)
add_update_codegen_reference_target(STENCIL hori_diff_stencil BACKENDS c++-naive cuda)
add_update_codegen_reference_target(STENCIL tridiagonal_solve_stencil BACKENDS c++-naive cuda)
add_update_codegen_reference_target(STENCIL update_dz_c BACKENDS c++-naive cuda)

# Unstructured stencils
add_update_codegen_reference_target(STENCIL ICON_laplacian_stencil BACKENDS c++-naive-ico)
add_update_codegen_reference_target(STENCIL simple_reduction_stencil BACKEND c++-naive-ico)
add_update_codegen_reference_target(STENCIL sparse_dimensions BACKEND c++-naive-ico)
add_update_codegen_reference_target(STENCIL tridiagonal_solve_unstructured BACKEND c++-naive-ico)
