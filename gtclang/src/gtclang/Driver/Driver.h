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

#ifndef GTCLANG_DRIVER_DRIVER_H
#define GTCLANG_DRIVER_DRIVER_H

#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "gtclang/Driver/Options.h"
#include "llvm/ADT/SmallVector.h"
#include <list>
#include <string>
#include <vector>

namespace gtclang {

struct ReturnValue {
  int ExitCode;
  std::shared_ptr<dawn::SIR> SIR;

  operator int() { return ExitCode; }
};

/// @brief Main driver of gtclang
/// @ingroup driver
struct Driver : public dawn::NonCopyable {

  /// @brief Run gtclang on the given arguments
  /// @returns The Stencil Intermediate Representation and an integer that is `0` on success, `1`
  /// otherwise
  static ReturnValue run(const llvm::SmallVectorImpl<const char*>& args);

  static bool isInitialized;
};

/// @brief Driver for the gtclang parser
/// @ingroup driver
std::shared_ptr<dawn::SIR> run(const std::string& fileName, ParseOptions options);

} // namespace gtclang

#endif
