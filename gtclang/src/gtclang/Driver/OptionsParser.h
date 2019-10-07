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

#ifndef GTCLANG_DRIVER_OPTIONSPARSER_H
#define GTCLANG_DRIVER_OPTIONSPARSER_H

#include "dawn/Support/NonCopyable.h"
#include "gtclang/Driver/Options.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gtclang {

/// @brief GTClang options parser
/// @ingroup driver
class OptionsParser : dawn::NonCopyable {
  using OptionsMap =
      std::unordered_map<std::string, std::function<bool(Options*, const char*, bool)>>;
  using OptionsAliasMap = std::unordered_map<std::string, std::vector<llvm::StringRef>>;

  Options* options_;
  OptionsMap optionsMap_;
  OptionsAliasMap optionsAliasMap_;

public:
  /// @brief Construct options parser with the `options` the result is stored to
  OptionsParser(Options* options);

  /// @brief Extract options of GTClang and prepare the options for the Clang Frontend
  ///
  /// @param args       Command-line options
  /// @param clangArgs  `args` without the GTClang options
  /// @return Returns `true` on success, `false` otherwise
  bool parse(const llvm::SmallVectorImpl<const char*>& args,
             llvm::SmallVectorImpl<const char*>& clangArgs);
};

} // namespace gtclang

#endif
