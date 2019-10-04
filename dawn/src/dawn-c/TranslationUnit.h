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

#ifndef DAWN_C_TRANSLATIONUNIT_H
#define DAWN_C_TRANSLATIONUNIT_H

#include "dawn-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dawn_c
 * @{
 */

/**
 * @brief Destroy the translation unit
 */
extern void dawnTranslationUnitDestroy(dawnTranslationUnit_t* translationUnit);

/**
 * @brief Get the necessary preprocessor defines
 *
 * @param[in]   translationUnit   Translation unit to use
 * @param[out]  ppDefines         Array of '\0' termianted strings of length `num` which contains
 *                                the preprocessor definitions required
 * @param[out]  size              Size of array `ppDefines`
 */
extern void dawnTranslationUnitGetPPDefines(const dawnTranslationUnit_t* translationUnit,
                                            char*** ppDefines, int* size);
/**
 * @brief Get the generated code of the stencil `name`
 *
 * @param[in]   translationUnit    Translation unit to use
 * @param[in]   name               Name of the stencil
 * @returns newly allocated '\0' terminated string of the generated code of stencil `name` (returns
 *          `NULL` if stencil `name` was not found)
 */
extern char* dawnTranslationUnitGetStencil(const dawnTranslationUnit_t* translationUnit,
                                           const char* name);

/**
 * @brief Get the generated code for the global variables
 *
 * @param[in]   translationUnit   Translation unit to use
 * @returns newly allocated '\0' terminated string of the generated code of the globals
 */
extern char* dawnTranslationUnitGetGlobals(const dawnTranslationUnit_t* translationUnit);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
