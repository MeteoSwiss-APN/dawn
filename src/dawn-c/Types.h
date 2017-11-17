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

#ifndef DAWN_C_TYPES_H
#define DAWN_C_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dawn_c
 * @{
 */

/**
 * @brief Supported types
 *
 * Note that booleans are represented as integers.
 */
enum DawnTypeKind {
  DK_Integer, /**< `int` */
  DK_Double,  /**< `double` */
  DK_Char     /**< `char` */
};

/**
 * @brief Refrence to the Options
 */
typedef struct {
  void* Impl;   /**< Pointer to the allocated dawn::Options */
  int OwnsData; /**< Ownership flag */
} dawnOptions_t;

/**
 * @brief Refrence to an entry in the Options map
 */
typedef struct {
  DawnTypeKind Type;  /**< Type of the option */
  size_t SizeInBytes; /**< Total size in bytes */
  void* Value;        /**< Pointer to the allocated memory of the value */
} dawnOptionsEntry_t;

/**
 * @brief Refrence to the Compiler
 */
typedef struct {
  void* Impl;   /**< Pointer to the allocated dawn::DawnCompiler */
  int OwnsData; /**< Ownership flag */
} dawnCompiler_t;

/** @} */

#ifdef __cplusplus
}
#endif

#endif
