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

#ifndef GTCLANG_UNITTEST_GTCLANG_H
#define GTCLANG_UNITTEST_GTCLANG_H

#include "gsl/Support/NonCopyable.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief Emulate invocation of GTClang from command-line
/// @ingroup unittest
class GTClang : gsl::NonCopyable {
public:
  /// @brief Run GTClang with given flags
  ///
  /// @return `true` on success, `false` otherwise
  static bool run(const std::vector<std::string>& gtclangFlags,
                  const std::vector<std::string>& clangFlags);
};

} // namespace gtclang

#endif
