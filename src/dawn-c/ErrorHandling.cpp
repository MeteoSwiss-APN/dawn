//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn-c/ErrorHandling.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static void dawnDefaultFatalErrorHandler(const char* reason) {
  std::fprintf(stderr, "dawn: ERROR: %s\n", reason);
  std::fflush(stderr);
  std::exit(1);
}

static dawnFatalErrorHandler_t FatalErrorHandler = dawnDefaultFatalErrorHandler;

void dawnInstallFatalErrorHandler(dawnFatalErrorHandler_t handler) {
  FatalErrorHandler = handler ? handler : dawnDefaultFatalErrorHandler;
}

void dawnFatalError(const char* reason) {
  assert(FatalErrorHandler);
  (*FatalErrorHandler)(reason);
}

static struct ErrorState {
  bool HasError = false;
  std::string ErrorMsg = "";
} errorState;

void dawnStateErrorHandler(const char* reason) {
  errorState.HasError = true;
  errorState.ErrorMsg = reason;
}

int dawnStateErrorHandlerHasError(void) { return errorState.HasError; }

char* dawnStateErrorHandlerGetErrorMessage(void) {
  std::size_t size = errorState.ErrorMsg.size() + 1;
  char* errorMessage = (char*)std::malloc(size * sizeof(char));
  std::memcpy(errorMessage, errorState.ErrorMsg.c_str(), size);
  return errorMessage;
}

void dawnStateErrorHandlerResetState(void) {
  errorState.HasError = false;
  errorState.ErrorMsg.clear();
}
