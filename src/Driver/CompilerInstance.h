//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _     _ _              _            _
//                        (_)   | | |            | |          | |
//               __ _ _ __ _  __| | |_ ___   ___ | |___    ___| | __ _ _ __   __ _
//              / _` | '__| |/ _` | __/ _ \ / _ \| / __|  / __| |/ _` | '_ \ / _` |
//             | (_| | |  | | (_| | || (_) | (_) | \__ \ | (__| | (_| | | | | (_| |
//              \__, |_|  |_|\__,_|\__\___/ \___/|_|___/  \___|_|\__,_|_| |_|\__, |
//               __/ |                                                        __/ |
//              |___/                                                        |___/
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GTCLANG_DRIVER_COMPILERINSTANCE_H
#define GTCLANG_DRIVER_COMPILERINSTANCE_H

#include "llvm/ADT/SmallVector.h"

namespace clang {
class CompilerInstance;
}

namespace gtclang {

/// @brief Create a CompilerInstance based on the commandline arguments, or `NULL` if there's an
/// error of some sort
///
/// @param args   Arguments passed to the Clang Frontend (may be modified)
///
/// @ingroup driver
clang::CompilerInstance* createCompilerInstance(llvm::SmallVectorImpl<const char*>& args);

} // namespace gtclang

#endif
