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

#ifndef GTCLANG_FRONTEND_PREPROCESSORACTION_H
#define GTCLANG_FRONTEND_PREPROCESSORACTION_H

#include "clang/Frontend/FrontendActions.h"

namespace gtclang {

class GTClangContext;

/// @brief Preprocessor-only action which replaces the enhanced with the pure gridtools clang DSL
/// @ingroup frontend
class GTClangPreprocessorAction : public clang::PreprocessOnlyAction {
  GTClangContext* context_;

public:
  GTClangPreprocessorAction(GTClangContext* context);

protected:
  void ExecuteAction();
};

} // namespace gtclang

#endif
