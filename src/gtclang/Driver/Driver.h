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

#include "gsl/Support/NonCopyable.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief Main driver of gtclang
/// @ingroup driver
struct Driver : public gsl::NonCopyable {

  /// @brief Run gtclang on the given arguments
  /// @returns `0` on success, `1` otherwise
  static int run(const llvm::SmallVectorImpl<const char*>& args);
};

} // namespace gtclang

#endif
