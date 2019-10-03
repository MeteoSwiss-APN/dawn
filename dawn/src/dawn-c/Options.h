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

#ifndef DAWN_C_OPTIONS_H
#define DAWN_C_OPTIONS_H

#include "dawn-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dawn_c
 * @{
 */

/*===------------------------------------------------------------------------------------------===*\
 *     OptionsEntry
\*===------------------------------------------------------------------------------------------===*/

/**
 * @brief Create an integer options entry
 */
extern dawnOptionsEntry_t* dawnOptionsEntryCreateInteger(int value);

/**
 * @brief Create a double options entry
 */
extern dawnOptionsEntry_t* dawnOptionsEntryCreateDouble(double value);

/**
 * @brief Create a string options entry
 *
 * Note that the string needs to be `\0` terminated
 */
extern dawnOptionsEntry_t* dawnOptionsEntryCreateString(const char* value);

/**
 * @brief Get a *copy* of the value of an entry which stores an integer
 * @returns newly allocated pointer to the value of the entry or `NULL` if the `entry` does no store
 *          an integer
 */
extern int* dawnOptionsEntryGetInteger(dawnOptionsEntry_t* entry);

/**
 * @brief Get a *copy* of the value of an entry which stores a double
 * @returns newly allocated pointer to the value of the entry or `NULL` if the `entry` does no store
 *          a double
 */
extern double* dawnOptionsEntryGetDouble(dawnOptionsEntry_t* entry);

/**
 * @brief Get a *copy* of the value of an entry which stores a string
 * @returns newly allocated pointer to the value of the entry or `NULL` if the `entry` does no store
 *          a string
 */
extern char* dawnOptionsEntryGetString(dawnOptionsEntry_t* entry);

/**
 * @brief Destroy the Options and deallocate all memory
 */
extern void dawnOptionsEntryDestroy(dawnOptionsEntry_t* entry);

/*===------------------------------------------------------------------------------------------===*\
 *     Options
\*===------------------------------------------------------------------------------------------===*/

/**
 * @brief Default construct the Options
 *
 * The entry needs to be destroyed via @ref dawnOptionsEntryDestroy.
 */
extern dawnOptions_t* dawnOptionsCreate(void);

/**
 * @brief Destroy the Options and deallocate all memory
 */
extern void dawnOptionsDestroy(dawnOptions_t* options);

/**
 * @brief Check of option `name` exists
 */
extern int dawnOptionsHas(const dawnOptions_t* options, const char* name);

/**
 * \brief Get the option `name`
 *
 * This function allocates a new `dawnOptionsEntry_t` which contains the type information and
 * value of the option `name`.
 *
 * The entry needs to be destroyed via @ref dawnOptionsEntryDestroy.
 *
 * @param name    Name of the options
 * @returns Newly allocated options entry or calls `dawnFatalError` if option does not exists
 */
extern dawnOptionsEntry_t* dawnOptionsGet(const dawnOptions_t* options, const char* name);

/**
 * @brief Set the option `name` to `value`
 */
extern void dawnOptionsSet(dawnOptions_t* options, const char* name,
                           const dawnOptionsEntry_t* value);

/**
 * @brief Convert to string
 *
 * The function will allocate a sufficiently large `char` buffer (using malloc()) which needs
 * be freed by the user using free().
 *
 * @return String representation of the options
 */
extern char* dawnOptionsToString(const dawnOptions_t* options);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
