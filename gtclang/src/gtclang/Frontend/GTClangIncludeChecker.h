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

#ifndef GTCLANG_FRONTEND_INCLUDEPROCESSOR_H
#define GTCLANG_FRONTEND_INCLUDEPROCESSOR_H

#include "clang/Frontend/FrontendActions.h"
#include "llvm/ADT/StringRef.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace gtclang {

using clang::StringRef;

/// @brief Ensures that input file has necessary gtclang includes needed for stencil DSL
/// @ingroup frontend
class GTClangIncludeChecker {
public:
  GTClangIncludeChecker();
  void Update(const std::string& sourceFile);
  void Restore();

protected:
  void ScanHeader(const llvm::SmallVector<StringRef, 100>& PPCodeLines, std::vector<std::string>& includes, std::vector<std::string>& namespaces);
  void WriteFile(const llvm::SmallVector<StringRef, 100>& PPCodeLines, std::vector<std::string>& includes, std::vector<std::string>& namespaces);

private:
  bool updated_;
  std::string sourceFile_;
};

} // namespace gtclang

#endif
