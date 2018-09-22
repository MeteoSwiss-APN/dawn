/*===----------------------------------------------------------------------------------*- C -*-===*\
 *                          _
 *                         | |
 *                       __| | __ ___      ___ ___
 *                      / _` |/ _` \ \ /\ / / '_  |
 *                     | (_| | (_| |\ V  V /| | | |
 *                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
 *
 *
 *  This file is distributed under the MIT License (MIT).
 *  See LICENSE.txt for details.
 *
\*===------------------------------------------------------------------------------------------===*/

#ifndef DAWN_C_DAWNCOMPILER_H
#define DAWN_C_DAWNCOMPILER_H

#include "dawn-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dawn_c
 * @{
 */

/**
 * @brief Report a diagnostics encountered during compilation
 *
 * This function will invoke the installed DiagnosticsHandler.
 *
 * @see dawnInstallDiagnosticsHandler
 */
extern void dawnReportDiagnostic(DawnDiagnosticsKind diag, int line, int column,
                                 const char* filename, const char* msg);

/**
 * @brief Diagnostics handler callback
 */
typedef void (*dawnDiagnosticsHandler_t)(DawnDiagnosticsKind diag, int line, int column,
                                         const char* filename, const char* msg);

/**
 * @brief Install a diagnostics handler
 *
 * By default, the diagnostics will be formatted and printed to `stderr`.
 *
 * @param handler   New diagnostics handler (or if `NULL` is passed the default handler will be
 *                  restored)
 */
extern void dawnInstallDiagnosticsHandler(dawnDiagnosticsHandler_t handler);

/**
 * @brief Run the compiler on the byte-string serialized SIR and return the generated code
 *
 * @param SIR         Byte string serialized data of the SIR
 * @param size        Size of the serialized SIR data
 * @param options     Options of the compilation (if `NULL` is passed the default options are used)
 * @param codeGenKind Code generation backend to use
 * @return Translation unit of the generated code or `NULL` on failure
 */
extern dawnTranslationUnit_t* dawnCompile(const char* SIR, size_t size,
                                          const dawnOptions_t* options);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
