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

#ifndef GTCLANG_FRONTEND_STENCILPARSER
#define GTCLANG_FRONTEND_STENCILPARSER

#include "dawn/SIR/SIR.h"
#include "gtclang/Frontend/ClangASTStmtResolver.h"
#include "gtclang/Frontend/Diagnostics.h"
#include "clang/AST/ASTFwd.h"
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace gtclang {

class GTClangContext;
class ClangASTExprResolver;
class GlobalVariableParser;

/// @brief Convert AST declaration of a stencil to SIR
/// @ingroup frontend
class StencilParser : dawn::NonCopyable {
public:
  /// @brief The type of stencil we are going to parse
  enum StencilKind { SK_Stencil, SK_StencilFunction };

  friend class ClangASTStmtResolver;

private:
  GTClangContext* context_;
  GlobalVariableParser& globalVariableParser_;

  /// Map of the parsed stencils and stencil functions
  std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::Stencil>> stencilMap_;
  std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::StencilFunction>> stencilFunctionMap_;

  /// Registered interval levels (to speedup lookup)
  std::unordered_map<std::string, int> customIntervalLevel_;

  /// Record of the currently parsed stencil or stencil function
  struct ParserRecord {
    ParserRecord(StencilKind kind) : CurrentKind(kind) {}

    /// Current kind i.e are we currently parsing a stencil or stencil function?
    StencilKind CurrentKind;

    /// The declaration statement of the current stencil or stencil function
    clang::CXXRecordDecl* CurrentCXXRecordDecl = nullptr;

    /// The current SIR stencil description (might be NULL and incomplete)
    dawn::sir::Stencil* CurrentStencil = nullptr;

    /// The current SIR stencil function descroption (might be NULL and incomplete)
    dawn::sir::StencilFunction* CurrentStencilFunction = nullptr;

    struct ArgDecl {
      ArgDecl(int index, const std::string& name, clang::FieldDecl* decl)
          : Index(index), Name(name), Decl(decl) {}

      int Index;              ///< Index of the argument
      std::string Name;       ///< Name of the argument
      clang::FieldDecl* Decl; ///< Clang AST declaration of the argument
    };

    /// Map of the arguments (a.k.a members) of the current stencil or stencil function
    std::unordered_map<std::string, ArgDecl> CurrentArgDeclMap;

    /// @brief Add an argument declaration
    void addArgDecl(const std::string& name, clang::FieldDecl* decl);
  };
  std::unique_ptr<ParserRecord> currentParserRecord_;

public:
  StencilParser(GTClangContext* context, GlobalVariableParser& globalVariableParser);

  /// @brief Parse a `gridtools::clang::stencil`
  void parseStencil(clang::CXXRecordDecl* recordDecl, const std::string& name);

  /// @brief Parse a `gridtools::clang::stencil_function`
  void parseStencilFunction(clang::CXXRecordDecl* recordDecl, const std::string& name);

  /// @brief Get the StencilMap
  const std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::Stencil>>& getStencilMap() const;

  /// @brief Get the StencilFunctionMap
  const std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::StencilFunction>>&
  getStencilFunctionMap() const;

  /// @brief Report a diagnostic
  clang::DiagnosticBuilder reportDiagnostic(clang::SourceLocation loc, Diagnostics::DiagKind kind);

  /// @brief Get stencil-function map entry by stencil-function `name` returns (NULL, NULL) if
  /// stencil function does not exist
  std::pair<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::StencilFunction>>
  getStencilFunctionByName(const std::string& name);

  /// @brief Check if stencil function with given `name` exists
  bool hasStencilFunction(const std::string& name) const;

  /// @brief Get the current parser record
  const ParserRecord* getCurrentParserRecord() const;

  /// @brief Get the GTClangContext
  GTClangContext* getContext() { return context_; }

  /// @brief Get the custom interval level `name`
  /// @returns Pair with first equal to `True` if a custom interval level with given `name` exists
  /// and second containing the level
  std::pair<bool, int> getCustomIntervalLevel(const std::string& name) const;

  /// @brief Register a parsed the custom interval level
  void setCustomIntervalLevel(const std::string& name, int level);

  /// @brief Check if the global variable `name` is registered
  bool isGlobalVariable(const std::string& name) const;

private:
  void parseStencilImpl(clang::CXXRecordDecl* recordDecl, const std::string& name,
                        StencilKind kind);

  /// @brief Parse a `storage` of a stencil
  void parseStorage(clang::FieldDecl* field);

  /// @brief Parse an argument of a stencil-function
  void parseArgument(clang::FieldDecl* arg);

  /// @brief Parse the `Do-Method` of a stencil
  void parseStencilDoMethod(clang::CXXMethodDecl* DoMethod);

  /// @brief Parse the `Do-Method` of a stencil-function
  void parseStencilFunctionDoMethod(clang::CXXMethodDecl* DoMethod);

  /// @brief Parse call to another stencil
  std::shared_ptr<dawn::sir::StencilCallDeclStmt> parseStencilCall(clang::CXXConstructExpr* stencilCall);

  /// @brief Parse a vertical-region
  std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>
  parseVerticalRegion(clang::CXXForRangeStmt* verticalRegionDecl);

  /// @brief Parse a decription of a boundary condition
  std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt>
  parseBoundaryCondition(clang::CXXConstructExpr* boundaryCondition);

  /// @brief Parses the MethodDeclaration that the preprocessor generates that contains all the
  /// boundary-condition statements
  std::vector<std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt>>
  parseBoundaryConditions(clang::CXXMethodDecl* allBoundaryConditions);

  /// @brief Resolve Clang AST statements by passing them to ClangASTResolver
  std::shared_ptr<dawn::sir::Stmt> resolveStmt(ClangASTExprResolver& clangASTResolver,
                                          clang::Stmt* stmt);

  /// @brief Get SourceLocation
  dawn::SourceLocation getLocation(clang::Decl* decl) const;
  dawn::SourceLocation getLocation(clang::Stmt* stmt) const;
};

} // namespace gtclang

#endif
