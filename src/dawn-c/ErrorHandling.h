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

#ifndef DAWN_C_ERRORHANDLING_H
#define DAWN_C_ERRORHANDLING_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup dawn_c
 * @{
 */

/**
 * @brief Report a fatal error
 *
 * This function will invoke the installed FatalErrorHandler.
 *
 * @see dawnInstallFatalErrorHandler
 */
extern void dawnFatalError(const char* reason);

/**
 * @brief Error handler callback
 */
typedef void (*dawnFatalErrorHandler_t)(const char* reason);

/**
 * @brief Install a fatal error handler
 *
 * By default, if dawn detects a fatal error it will emit the last error string to stderr and
 * call exit(1).
 *
 * This may not be appropriate in some contexts. For example, the Python module might want to
 * translate the errror into an exception. This function allows you to install a callback that will
 * be invoked after a fatal error occurred.
 *
 * @param handler   New error handler (or if `NULL` is passed the default handler will be restored)
 */
extern void dawnInstallFatalErrorHandler(dawnFatalErrorHandler_t handler);

/**
 * @brief Store the the current state of the error which can be queried via
 * @ref dawnStateErrorHandlerHasError as well as @ref dawnStateErrorHandlerGetErrorMessage
 *
 * This error handler is **not** thread-safe.
 */
extern void dawnStateErrorHandler(const char* reason);

/**
 * @brief Check the current error state
 *
 * This function requires to set the ErrorHandler to  @ref dawnStateErrorHandler. To obtain the
 * associated error message, use @ref dawnStateErrorHandlerGetErrorMessage.
 *
 * @return 1 if there was an error, 0 otherwise
 */
extern int dawnStateErrorHandlerHasError(void);

/**
 * @brief Query the current error state
 *
 * This function requires to set the ErrorHandler to @ref dawnStateErrorHandler.
 *
 * @return newly allocated `char*` with the current error message
 */
extern char* dawnStateErrorHandlerGetErrorMessage(void);

/**
 * @brief Reset the current error state
 */
extern void dawnStateErrorHandlerResetState(void);

/** @} */

#ifdef __cplusplus
}
#endif

#endif
