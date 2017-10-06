//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _                   
//                        | |     | |                  
//                    __ _| |_ ___| | __ _ _ __   __ _ 
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/ 
//
//
//  This file is distributed under the MIT License (MIT). 
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

// clang-format off

// General 
DIAG(note_previous_definition, clang::DiagnosticIDs::Note, "previous definition is here")
DIAG(note_declaration, clang::DiagnosticIDs::Note, "declaration is here")
DIAG(err_invalid_option, clang::DiagnosticIDs::Error, "invalid value of option '%0', possible values are '%1'")

// File system
DIAG(err_fs_error, clang::DiagnosticIDs::Error, "file system error: %0")

// DoMethod
DIAG(err_do_method_missing , clang::DiagnosticIDs::Error, "missing Do-method for stencil declaration '%0'")
DIAG(err_do_method_ill_formed, clang::DiagnosticIDs::Error, "ill-formed Do-method")
DIAG(err_do_method_illegal_stmt, clang::DiagnosticIDs::Error, "illegal statement in Do-method")
DIAG(err_do_method_non_local_var, clang::DiagnosticIDs::Error, "reference to non-local variable '%0' inside Do-method")
DIAG(err_do_method_invalid_type_of_local_var, clang::DiagnosticIDs::Error, "invalid type '%0' of local variable '%1'")
DIAG(err_do_method_non_const_array_type, clang::DiagnosticIDs::Error, "non-constant array type '%0' of local variable '%0'")
DIAG(err_do_method_invalid_expr_if_cond, clang::DiagnosticIDs::Error, "invalid expression for condition in if-statement")
DIAG(err_do_method_nested_vertical_region, clang::DiagnosticIDs::Error, "invalid vertical region inside vertical region")
DIAG(err_do_method_var_shadowing, clang::DiagnosticIDs::Error, "declaration of local variable '%0' shadows argument of surrounding %1")

// Globals
DIAG(err_globals_multiple_definition, clang::DiagnosticIDs::Error, "'globals' can only be defined once")
DIAG(err_globals_invalid_type, clang::DiagnosticIDs::Error, "invalid type '%0' of global variable '%1'")
DIAG(err_globals_invalid_default_value, clang::DiagnosticIDs::Error, "invalid type '%0' of default value of global variable '%1'")
DIAG(err_globals_config, clang::DiagnosticIDs::Error, "config error: %0")
DIAG(note_globals_allowed_types, clang::DiagnosticIDs::Note, "only 'bool', 'int', 'double' or 'std::string' are allowed")

// BoundaryCondition
DIAG(err_boundary_condition_invalid_functor, clang::DiagnosticIDs::Error, "invalid functor '%0' in boundary condition: %1")
DIAG(err_boundary_condition_invalid_arg, clang::DiagnosticIDs::Error, "invalid argument '%0' in boundary condition")
DIAG(err_boundary_condition_invalid_type, clang::DiagnosticIDs::Error, "invalid type '%0' of argument '%1' in boundary condition")
DIAG(note_boundary_condition_only_storage_allowed, clang::DiagnosticIDs::Note, "only 'gridtools::clang::storage' is allowed")

// StencilFunction
DIAG(err_stencilfun_do_method_invalid_num_arg, clang::DiagnosticIDs::Error, "Do-method specialization requires exactly 2 arguments")
DIAG(err_stencilfun_do_method_invalid_range_keyword, clang::DiagnosticIDs::Error, "invalid parameter '%0' in Do-Method specialization only 'k_from' and 'k_to' are allowed")
DIAG(err_stencilfun_do_method_missing_interval, clang::DiagnosticIDs::Error, "missing interval specification of parameter '%0' (default argument required)")
DIAG(err_stencilfun_vertical_region, clang::DiagnosticIDs::Error, "invalid vertical region in stencil function")
DIAG(err_stencilfun_invalid_argument_type, clang::DiagnosticIDs::Error, "invalid type '%0' of stencil function argument '%1'") 
DIAG(err_stencilfun_invalid_argument, clang::DiagnosticIDs::Error, "argument '%0' is not a valid argument of the enclosing stencil function")
DIAG(err_stencilfun_expression_in_arg_list, clang::DiagnosticIDs::Error, "expression in argument list of stencil function is not supported")
DIAG(err_stencilfun_invalid_call, clang::DiagnosticIDs::Error, "no matching stencil function for call to '%0'")
DIAG(note_stencilfun_not_viable, clang::DiagnosticIDs::Note, "candidate stencil function not viable: %0")
DIAG(note_stencilfun_argument_declared_here, clang::DiagnosticIDs::Note, "argument declared here")

// StencilCall
DIAG(err_stencilcall_invalid_call, clang::DiagnosticIDs::Error, "no matching stencil for call to '%0'")
DIAG(err_stencilcall_invalid_argument_type, clang::DiagnosticIDs::Error, "invalid type '%0' of stencil argument '%1'") 
DIAG(note_stencilcall_only_storage_allowed, clang::DiagnosticIDs::Note, "only 'gridtools::clang::storage' is allowed")
DIAG(note_stencilcall_not_viable, clang::DiagnosticIDs::Note, "candidate stencil not viable: %0")

// Stencil
DIAG(err_stencil_invalid_storage_decl, clang::DiagnosticIDs::Error, "invalid type '%0' of stencil argument '%1'")
DIAG(note_only_storages_allowed, clang::DiagnosticIDs::Note, "only 'gridtools::clang::storage' or 'gridtools::clang::temporary_storage' are allowed as stencil arguments")

// Interval
DIAG(err_interval_invalid_type, clang::DiagnosticIDs::Error, "invalid type '%0' for vertical region declaration")
DIAG(err_interval_not_constexpr, clang::DiagnosticIDs::Error, "offset of vertical region is not a constant expression")
DIAG(err_interval_custom_missing_init, clang::DiagnosticIDs::Error, "missing initialization for custom interval '%0'")
DIAG(err_interval_custom_not_constexpr, clang::DiagnosticIDs::Error, "offset of built-in interval '%0' is not a constant expression")
DIAG(err_interval_custom_not_builtin, clang::DiagnosticIDs::Error, "invalid built-in interval '%0'")
DIAG(note_only_builtin_interval_allowed, clang::DiagnosticIDs::Note, "only 'gridtools::clang::k_start' or 'gridtools::clang::k_end' are allowed built-in intervals")

// Index
DIAG(err_index_not_constexpr, clang::DiagnosticIDs::Error, "offset of index is not a constant expression")
DIAG(err_index_illegal_argument, clang::DiagnosticIDs::Error, "index '%0' is not a valid argument of the enclosing stencil function")
DIAG(err_index_invalid_type, clang::DiagnosticIDs::Error, "invalid type '%0' of index '%1'")

// clang-format on
