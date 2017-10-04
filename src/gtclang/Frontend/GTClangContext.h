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

#ifndef GTCLANG_FRONTEND_GTCLANGCONTEXT
#define GTCLANG_FRONTEND_GTCLANGCONTEXT

#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "gtclang/Driver/Options.h"
#include "gtclang/Frontend/Diagnostics.h"
#include <memory>
#include <unordered_map>

namespace clang {
class ASTContext;
}

namespace gtclang {

/// @brief Context of the GTClang tool
/// @ingroup frontend
class GTClangContext : dawn::NonCopyable {
  std::unique_ptr<Options> options_;
  std::unique_ptr<Diagnostics> diagnostics_;

  // Map of attributes for stencil and stencil-functions
  std::unordered_map<std::string, dawn::sir::Attr> stencilNameToAttributeMap_;

  // Raw points are always non-owning
  clang::ASTContext* astContext_;

public:
  GTClangContext();

  /// @name Get configuration options parsed from command-line
  /// @{
  Options& getOptions();
  const Options& getOptions() const;
  /// @}

  /// @name Get diagnostics
  /// @{
  Diagnostics& getDiagnostics();
  const Diagnostics& getDiagnostics() const;
  void setDiagnostics(clang::DiagnosticsEngine* diags);
  bool hasDiagnostics() const;
  /// @}

  /// @name Get/Set AST context
  /// @{
  clang::ASTContext& getASTContext();
  const clang::ASTContext& getASTContext() const;
  void setASTContext(clang::ASTContext* astContext);
  /// @}

  /// @name Get SourceManager
  /// @{
  clang::SourceManager& getSourceManager();
  const clang::SourceManager& getSourceManager() const;
  /// @}

  /// @name Get Clang DiagnosticsEngine
  /// @{
  clang::DiagnosticsEngine& getDiagnosticsEngine();
  const clang::DiagnosticsEngine& getDiagnosticsEngine() const;
  /// @}

  /// @brief Get/Set attribute of stencil and stencil functions
  ///
  /// If attribute was not set for the given stencil or stencil function, the default constructed
  /// attribute is returned.
  ///
  /// @see dawn::sir::Attr
  /// @{
  dawn::sir::Attr getStencilAttribute(const std::string& name) const;
  void setStencilAttribute(const std::string& name, dawn::sir::Attr attr);
  /// @}
};

} // namespace gtclang

#endif
