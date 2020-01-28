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

#ifndef GTCLANG_UNITTEST_IRSPLITTER_H
#define GTCLANG_UNITTEST_IRSPLITTER_H

#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "gtclang/Unittest/GTClang.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief Emulate invocation of GTClang from command-line
/// @ingroup unittest
class IRSplitter : dawn::NonCopyable {
public:
  /// @brief Run GTClang with given flags
  ///
  /// @return a pair of a shared pointer to the SIR and a boolean `true` on success, `false`
  /// otherwise
  void split(const std::string& dslFile);
};

} // namespace gtclang

#endif
