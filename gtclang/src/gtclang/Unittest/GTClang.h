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

#ifndef GTCLANG_UNITTEST_GTCLANG_H
#define GTCLANG_UNITTEST_GTCLANG_H

#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief Emulate invocation of GTClang from command-line
/// @ingroup unittest
class GTClang : dawn::NonCopyable {
public:
  /// @brief Run GTClang with given flags
  ///
  /// @return a pair of a shared pointer to the SIR and a boolean `true` on success, `false`
  /// otherwise
  static std::pair<bool, std::shared_ptr<dawn::SIR>>
  run(const std::vector<std::string>& gtclangFlags, const std::vector<std::string>& clangFlags);
};

} // namespace gtclang

#endif
