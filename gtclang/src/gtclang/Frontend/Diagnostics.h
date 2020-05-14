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

#ifndef GTCLANG_FRONTEND_DIAGNOSTICS
#define GTCLANG_FRONTEND_DIAGNOSTICS

#include "dawn/Support/NonCopyable.h"
#include "clang/Basic/Diagnostic.h"

namespace gtclang {

/// @brief Handling of GTClang diagnostics
/// @ingroup frontend
class Diagnostics : dawn::NonCopyable {
public:
  enum DiagKind {
#define DIAG(ENUM, LEVEL, DESC) ENUM,
#include "gtclang/Frontend/DiagnosticsKind.inc"
    num_diags
  };
#undef DIAG

  /// @brief Register all GTClang diagnostics within the `DiagnosticsEngine`
  Diagnostics(clang::DiagnosticsEngine* diags);

  /// @brief Issue a diagnostics message to the client
  clang::DiagnosticBuilder report(clang::SourceLocation loc, DiagKind kind);
  clang::DiagnosticBuilder report(DiagKind kind) {
    return this->report(clang::SourceLocation(), kind);
  }
  clang::DiagnosticBuilder report(clang::SourceRange range, DiagKind kind);

  /// @brief Get the Clang diagnostics engine
  clang::DiagnosticsEngine& getDiagnosticsEngine() { return *diags_; }
  const clang::DiagnosticsEngine& getDiagnosticsEngine() const { return *diags_; }

  /// @brief Directly report to the diagnostics engine `diag`
  static void reportRaw(clang::DiagnosticsEngine& diag, clang::SourceLocation loc,
                        clang::DiagnosticIDs::Level level, const std::string& msg);

private:
  clang::DiagnosticsEngine* diags_;
};

} // namespace gtclang

#endif
