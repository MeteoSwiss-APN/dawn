##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _
##                        | |     | |
##                    __ _| |_ ___| | __ _ _ __   __ _
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

set(dummy_source ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/dummy.cpp)
file(WRITE ${dummy_source} "int main() { return 0; }")

add_subdirectory(CodeGen)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

function(add_integration_test source)
string(REGEX REPLACE "\\.[^.]*$" "" source_no_ext ${source})
set(test_name GTClang::Integration::${source_no_ext})
add_test(NAME ${test_name}
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/integration-test/tester.py $<TARGET_FILE:gtclang> ${CMAKE_CURRENT_LIST_DIR}/${source})
set_property(TEST ${test_name} APPEND PROPERTY DEPENDS gtclang)
endfunction()

add_integration_test(Accesses/Simple01.cpp)
add_integration_test(Accesses/Simple02.cpp)
add_integration_test(Accesses/Simple03.cpp)
add_integration_test(Accesses/Simple04.cpp)

add_integration_test(Accesses/StencilFunction01.cpp)
add_integration_test(Accesses/StencilFunction02.cpp)
add_integration_test(Accesses/StencilFunction03.cpp)
add_integration_test(Accesses/StencilFunction04.cpp)
add_integration_test(Accesses/StencilFunction05.cpp)
add_integration_test(Accesses/StencilFunction06.cpp)
add_integration_test(Accesses/StencilFunction07.cpp)

add_integration_test(Diagnostics/err_boundary_condition_invalid_type.cpp)
add_integration_test(Diagnostics/err_do_method_invalid_expr_if_cond.cpp)
add_integration_test(Diagnostics/err_do_method_invalid_type_of_local_var.cpp)
add_integration_test(Diagnostics/err_do_method_missing.cpp)
add_integration_test(Diagnostics/err_do_method_non_local_var.cpp)
add_integration_test(Diagnostics/err_do_method_var_shadowing.cpp)
add_integration_test(Diagnostics/err_globals_invalid_default_value.cpp)
add_integration_test(Diagnostics/err_globals_invalid_type.cpp)
add_integration_test(Diagnostics/err_index_not_constexpr.cpp)
add_integration_test(Diagnostics/err_interval_custom_missing_init.cpp)
add_integration_test(Diagnostics/err_interval_custom_not_builtin.cpp)
add_integration_test(Diagnostics/err_interval_custom_not_constexpr.cpp)
add_integration_test(Diagnostics/err_interval_invalid_type.cpp)
add_integration_test(Diagnostics/err_interval_offset_not_constexpr.cpp)
add_integration_test(Diagnostics/err_invalid_pragma.cpp)
add_integration_test(Diagnostics/err_stencil_invalid_storage_decl.cpp)
add_integration_test(Diagnostics/err_stencilcall_invalid_call_01.cpp)
add_integration_test(Diagnostics/err_stencilcall_invalid_call_02.cpp)
add_integration_test(Diagnostics/err_stencilfun_do_method_invalid_num_arg.cpp)
add_integration_test(Diagnostics/err_stencilfun_do_method_invalid_range_keyword.cpp)
add_integration_test(Diagnostics/err_stencilfun_invalid_argument.cpp)
add_integration_test(Diagnostics/err_stencilfun_invalid_argument_type.cpp)
add_integration_test(Diagnostics/err_stencilfun_invalid_call.cpp)
add_integration_test(Diagnostics/err_stencilfun_vertical_region.cpp)

add_integration_test(IIRSerializer/AccessTest.cpp)
add_integration_test(IIRSerializer/CopyTest.cpp)
add_integration_test(IIRSerializer/NestedStencils.cpp)
add_integration_test(IIRSerializer/StencilFnCallTest.cpp)

add_integration_test(PassFieldVersioning/RaceCondition01.cpp)
add_integration_test(PassFieldVersioning/RaceCondition02.cpp)
add_integration_test(PassFieldVersioning/RaceCondition03.cpp)
add_integration_test(PassFieldVersioning/VersioningTest01.cpp)
add_integration_test(PassFieldVersioning/VersioningTest02.cpp)
add_integration_test(PassFieldVersioning/VersioningTest03.cpp)
add_integration_test(PassFieldVersioning/VersioningTest04.cpp)
add_integration_test(PassFieldVersioning/VersioningTest05.cpp)
add_integration_test(PassFieldVersioning/VersioningTest06.cpp)
add_integration_test(PassFieldVersioning/VersioningTest07.cpp)

add_integration_test(PassMultiStageSplitter/SplitterTest01.cpp)
add_integration_test(PassMultiStageSplitter/SplitterTest02.cpp)
add_integration_test(PassMultiStageSplitter/SplitterTest03.cpp)
add_integration_test(PassMultiStageSplitter/SplitterTest04.cpp)
add_integration_test(PassMultiStageSplitter/SplitterTest05.cpp)

add_integration_test(PassPreprocessor/BracketAccessTest.cpp)
add_integration_test(PassPreprocessor/DoMethodTest.cpp)
add_integration_test(PassPreprocessor/GlobalsTest.cpp)
add_integration_test(PassPreprocessor/StructTest.cpp)
add_integration_test(PassPreprocessor/VarDeclarationsTest.cpp)
add_integration_test(PassPreprocessor/VerticalRegionTest.cpp)

# add_integration_test(PassSetBoundaryCondition/NoSplit.cpp)
# add_integration_test(PassSetBoundaryCondition/TwoFieldSplit.cpp)

add_integration_test(PassSetCaches/IJCacheTest01.cpp)
add_integration_test(PassSetCaches/IJCacheTest02.cpp)
add_integration_test(PassSetCaches/KCacheTest01.cpp)
add_integration_test(PassSetCaches/KCacheTest01b.cpp)
add_integration_test(PassSetCaches/KCacheTest02.cpp)
add_integration_test(PassSetCaches/KCacheTest02b.cpp)
add_integration_test(PassSetCaches/KCacheTest03.cpp)
add_integration_test(PassSetCaches/KCacheTest04.cpp)
add_integration_test(PassSetCaches/KCacheTest05.cpp)
add_integration_test(PassSetCaches/KCacheTest06.cpp)

add_integration_test(PassSetNonTempCaches/MultipleCaches_1.cpp)
add_integration_test(PassSetNonTempCaches/NoCache_1.cpp)
add_integration_test(PassSetNonTempCaches/NoCache_2.cpp)

add_integration_test(PassStageMerger/Test01.cpp)
add_integration_test(PassStageMerger/Test02.cpp)
add_integration_test(PassStageMerger/Test03.cpp)
add_integration_test(PassStageMerger/Test04.cpp)
add_integration_test(PassStageMerger/Test05.cpp)
add_integration_test(PassStageMerger/Test06.cpp)
add_integration_test(PassStageMerger/Test07.cpp)

add_integration_test(PassStageReordering/Test01.cpp)
add_integration_test(PassStageReordering/Test02.cpp)
add_integration_test(PassStageReordering/Test03.cpp)
add_integration_test(PassStageReordering/Test04.cpp)
add_integration_test(PassStageReordering/Test05.cpp)
add_integration_test(PassStageReordering/Test06.cpp)
add_integration_test(PassStageReordering/Test07.cpp)

add_integration_test(PassStageSplitter/SplitterTest01.cpp)
add_integration_test(PassStageSplitter/SplitterTest02.cpp)
add_integration_test(PassStageSplitter/SplitterTest03.cpp)
add_integration_test(PassStageSplitter/SplitterTest04.cpp)
add_integration_test(PassStageSplitter/SplitterTest05.cpp)
add_integration_test(PassStageSplitter/SplitterTest06.cpp)
add_integration_test(PassStageSplitter/SplitterTest07.cpp)

add_integration_test(PassTemporaryMerger/MergeTest01.cpp)
add_integration_test(PassTemporaryMerger/MergeTest02.cpp)
add_integration_test(PassTemporaryMerger/MergeTest03.cpp)
add_integration_test(PassTemporaryMerger/MergeTest04.cpp)
add_integration_test(PassTemporaryMerger/MergeTest05.cpp)

add_integration_test(PassTemporaryType/DemoteTest01.cpp)
add_integration_test(PassTemporaryType/PromoteTest01.cpp)
add_integration_test(PassTemporaryType/PromoteTest02.cpp)
add_integration_test(PassTemporaryType/PromoteTest03.cpp)
add_integration_test(PassTemporaryType/PromoteTest04.cpp)
add_integration_test(PassTemporaryType/PromoteTest05.cpp)

add_integration_test(Regression/globals_dependencies_issue401.cpp)

add_integration_test(SIR/AccessTest.cpp)
add_integration_test(SIR/CopyTest.cpp)
add_integration_test(SIR/StencilFnCallTest.cpp)
