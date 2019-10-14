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

#include "gtclang/Frontend/ClangASTExprResolver.h"
#include "dawn/SIR/AST.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/StringUtil.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/StencilParser.h"
#include "gtclang/Support/ASTUtils.h"
#include "gtclang/Support/ClangCompat/EvalResult.h"
#include "gtclang/Support/ClangCompat/SourceLocation.h"
#include "gtclang/Support/Logger.h"
#include "gtclang/Support/StringUtil.h"
#include "clang/AST/AST.h"
#include <sstream>
#include <stack>

namespace gtclang {

//===------------------------------------------------------------------------------------------===//
//     FunctionResolver
//===------------------------------------------------------------------------------------------===//

/// @brief State of the currently parsed c++ or stencil function
class FunctionResolver {
  enum ArgumentIndex { AK_Field = 0, AK_Direction, AK_Offset, AK_Unknown };
  enum FunctionKind { FK_CXXFunction, FK_StencilFunction };
  static const std::array<const char*, 3> TypeStr;

  ClangASTExprResolver* resolver_;

  /// Current scope of the function
  struct Scope {

    /// Type of function
    FunctionKind Kind;

    /// SIR description of the stencil function (primarly used to get the number/type of arguments)
    std::shared_ptr<dawn::sir::StencilFunction> SIRStencilFunction = nullptr;

    /// AST node of the C++ or stencil function i.e the result of the parsing
    std::shared_ptr<dawn::sir::FunCallExpr> DAWNFunCallExpr = nullptr;

    /// Clang AST node of the call to the stencil function (e.g `avg(i+1, u)`)
    clang::CXXConstructExpr* ClangCXXConstructExpr = nullptr;

    /// Clang AST node of the declaration of the stencil function (e.g `struct avg { ... }`)
    clang::CXXRecordDecl* ClangCXXRecordDecl = nullptr;

    /// Skip the next calls as they are consumed by an Unary, Binary or Ternary Expression in
    /// the argument list
    int ArgumentsToSkip = 0;

    /// Check if we already reported an error for this stencil function call
    bool DiagnosticIssued = false;

    /// Check if we are a nested stencil function (i.e called in the argument list of another
    /// stencil function)
    bool NestedStencilFunction = false;

    /// Did we got referenced by a `CXXConstructExpr` or `CXXTemporaryObjectExpr`?
    /// (See checkAndSetReferenced())
    bool Referenced = false;
  };

  std::stack<std::unique_ptr<Scope>> scope_;

public:
  FunctionResolver(ClangASTExprResolver* resolver) : resolver_(resolver) {}

  /// @brief Create a new stencil function
  bool push(clang::CXXConstructExpr* expr) {
    scope_.push(std::make_unique<Scope>());
    scope_.top()->Kind = FK_StencilFunction;

    std::string callee = getClassNameFromConstructExpr(expr);

    scope_.top()->DAWNFunCallExpr =
        std::make_shared<dawn::sir::StencilFunCallExpr>(callee, resolver_->getSourceLocation(expr));
    scope_.top()->ClangCXXConstructExpr = expr;

    auto stencilFunPair = resolver_->getParser()->getStencilFunctionByName(callee);

    scope_.top()->ClangCXXRecordDecl = stencilFunPair.first;
    scope_.top()->SIRStencilFunction = stencilFunPair.second;

    scope_.top()->NestedStencilFunction = scope_.size() > 1;
    scope_.top()->Referenced = clang::isa<clang::CXXTemporaryObjectExpr>(expr);

    // Stencil function does not exist
    if(!scope_.top()->SIRStencilFunction) {
      reportError();
      return false;
    }
    return true;
  }

  /// @brief Create a new c++ function
  bool push(clang::CallExpr* expr) {
    scope_.push(std::make_unique<Scope>());
    scope_.top()->Kind = FK_CXXFunction;

    DAWN_ASSERT_MSG(expr->getDirectCallee(), "only C-function calls are supported");
    scope_.top()->DAWNFunCallExpr = std::make_shared<dawn::sir::FunCallExpr>(
        expr->getDirectCallee()->getQualifiedNameAsString(), resolver_->getSourceLocation(expr));

    return true;
  }

  /// @brief Reset the resolver
  void reset() {
    while(!scope_.empty())
      scope_.pop();
  }

  /// @brief Check if stencil function with given `name` exists
  bool hasStencilFunction(const std::string& name) {
    return resolver_->getParser()->hasStencilFunction(name);
  }

  /// @brief Skip the next `N` arguments to the function (these are consumed by expressions
  /// like BinaryOperator etc..)
  void skipNextArguments(int N) { scope_.top()->ArgumentsToSkip += N; }

  /// @brief Add an argument to the current stencil function
  std::shared_ptr<dawn::sir::Expr> addArgument(clang::Expr* expr,
                                               const std::shared_ptr<dawn::sir::Expr>& arg) {
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    DAWN_ASSERT(isActive() && scope_.top()->DAWNFunCallExpr);

    if(scope_.top()->ArgumentsToSkip != 0) {
      scope_.top()->ArgumentsToSkip--;
      return arg;
    }

    scope_.top()->DAWNFunCallExpr->getArguments().emplace_back(arg);

    if(getCurrentKind() == FK_StencilFunction)
      checkTypeOfArgumentMatches(expr, arg.get());
    return arg;
  }

  /// @brief Check if we are currently parsing a function (C++ function or stencil function)
  bool isActive() const { return !scope_.empty(); }

  /// @brief Check if we are currently parsing a stencil function
  bool isActiveOnStencilFunction() const {
    return isActive() && getCurrentKind() == FK_StencilFunction;
  }

  /// @brief Nested stencil functions are constructed from `CXXTemporaryObjectExpr` instead of plain
  /// `CXXConstructExpr`.
  /// Consider the following example:
  ///
  /// @code
  ///   avg(i+1, avg(j+1, u));
  ///    ^        ^
  ///    1        2
  /// @endcode
  ///
  /// (1) will be constructed when we encounter the first `CXXConstructExpr`. Traversing down the
  /// tree (1) will be refrenced again as a `CXXTemporaryObjectExpr` or another `CXXConstructExpr`
  /// (quirk of the C++ language..). When this happens, we don't want to create it again (as it
  /// already exists) and preceed by traversing the arguments (see ClangASTResolver::resolve(
  /// clang::CXXConstructExpr*)) the second if branch.  After parsing the first argument, we will
  /// encounter (2), which is given as well as a `CXXConstructExpr` or a `CXXTemporaryObjectExpr`,
  /// this time we want to create a new stencil function! To be able to distinguish those two cases
  /// we need to record if we were created as in nested scope and if this is the first time we are
  /// referenced.
  ///
  /// @return `true` if we are @b not a nested stencil function and have not yet been referenced
  bool checkAndSetReferenced() {
    if(scope_.empty())
      return false;

    if(getCurrentKind() != FK_StencilFunction)
      return false;

    if(!scope_.top()->NestedStencilFunction && !scope_.top()->Referenced) {
      scope_.top()->Referenced = true;
      return true;
    } else {
      return false;
    }
  }

  /// @brief Get the StencilFunCallExpr and pop the current stencil function
  std::shared_ptr<dawn::sir::FunCallExpr> pop() {
    auto expr = scope_.top()->DAWNFunCallExpr;
    scope_.pop();
    return expr;
  }

  /// @brief Check if number of arguments match
  void checkNumOfArgumentsMatch() {
    DAWN_ASSERT(isActive() && scope_.top()->DAWNFunCallExpr);
    DAWN_ASSERT(getCurrentKind() == FK_StencilFunction);

    std::size_t requiredArgs = scope_.top()->SIRStencilFunction->Args.size();
    std::size_t parsedArgs = scope_.top()->DAWNFunCallExpr->getArguments().size();

    if(parsedArgs != requiredArgs) {
      if(!reportError())
        return;

      std::stringstream ss;
      ss << "requires " << requiredArgs << " argument" << (requiredArgs == 1 ? "" : "s") << ", but "
         << parsedArgs << " " << (parsedArgs == 1 ? "was" : "were") << " provided";
      reportNote(ss.str());
    }
  }

private:
  /// @brief Get the current kind of function
  FunctionKind getCurrentKind() const {
    DAWN_ASSERT(isActive());
    return scope_.top()->Kind;
  }

  /// @brief Report a note
  void reportNote(const std::string& msg) {
    resolver_->reportDiagnostic(scope_.top()->ClangCXXRecordDecl->getLocation(),
                                Diagnostics::note_stencilfun_not_viable)
        << msg;
  }

  /// @brief Report an error
  bool reportError() {
    if(!scope_.top()->DiagnosticIssued) {

      auto* ClangCXXConstructExpr = scope_.top()->ClangCXXConstructExpr;
      resolver_->reportDiagnostic(ClangCXXConstructExpr->getLocation(),
                                  Diagnostics::err_stencilfun_invalid_call)
          << getClassNameFromConstructExpr(ClangCXXConstructExpr)
          << ClangCXXConstructExpr->getSourceRange();
      scope_.top()->DiagnosticIssued = true;
      return true;
    }
    return false;
  }

  /// @brief Check if types of arguments match
  void checkTypeOfArgumentMatches(clang::Expr* expr, dawn::sir::Expr* parsedArg) {
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    int curIndex = scope_.top()->DAWNFunCallExpr->getArguments().size() - 1;

    // To many arguments.. we diagnose that later
    if(curIndex >= scope_.top()->SIRStencilFunction->Args.size())
      return;

    ArgumentIndex parsedIdx = getIndexFromASTArg(parsedArg);
    ArgumentIndex requiredIdx =
        getIndexFromSIRArg(scope_.top()->SIRStencilFunction->Args[curIndex].get());

    // Everything is fine ...
    if(parsedIdx == requiredIdx)
      return;
    else {
      // It is fine to pass a direction (e.g `i`) as an offset
      if(requiredIdx == AK_Offset && parsedIdx == AK_Direction)
        return;

      // We have an actual mismatch!
      if(!reportError())
        return;

      std::stringstream ss;
      std::string parsedType =
          (parsedIdx == AK_Unknown ? expr->getType().getAsString() : TypeStr[parsedIdx]);

      ss << "no known conversion from '" << parsedType << "' to '" << TypeStr[requiredIdx]
         << "' for " << dawn::decimalToOrdinal(curIndex + 1) << " argument";
      reportNote(ss.str());
    }
  }

  /// @brief Convert an SIR stencil function argument to the matching index in `TypeStr`
  static ArgumentIndex getIndexFromSIRArg(dawn::sir::StencilFunctionArg* arg) {
    if(dawn::isa<dawn::sir::Field>(arg))
      return AK_Field;
    else if(dawn::isa<dawn::sir::Direction>(arg))
      return AK_Direction;
    else // == dawn::sir::Offset
      return AK_Offset;
  };

  /// @brief Convert an AST stencil function argument to the matching index in `TypeStr`
  static ArgumentIndex getIndexFromASTArg(dawn::sir::Expr* expr) {
    if(dawn::isa<dawn::sir::FieldAccessExpr>(expr) ||
       dawn::isa<dawn::sir::StencilFunCallExpr>(expr))
      return AK_Field;

    dawn::sir::StencilFunArgExpr* arg = dawn::dyn_cast<dawn::sir::StencilFunArgExpr>(expr);
    if(!arg)
      return AK_Unknown;

    if(arg->getOffset() != 0)
      return AK_Offset;
    else
      return AK_Direction;
  }
};

const std::array<const char*, 3> FunctionResolver::TypeStr = {
    {"gridtools::clang::storage", "gridtools::clang::direction", "gridtools::clang::offset"}};

namespace {

//===------------------------------------------------------------------------------------------===//
//     StorageResolver
//===------------------------------------------------------------------------------------------===//

/// @brief Parse a field access (e.g `u(i, j, k)` or simply `u` or `u(dir)` where `dir` is a
/// directional or offset parameter of an enclosing stencil function
class StorageResolver {
  enum CurrentArugmentKind { CK_Dimension, CK_Direction, CK_Offset };

  ClangASTExprResolver* resolver_;

  int curDim_;
  int curArg_;
  CurrentArugmentKind currentArgumentKind_;

  // FieldAccessExpr parameter
  std::string name_;
  dawn::Array3i offset_;
  dawn::Array3i argumentMap_;
  dawn::Array3i argumentOffset_;
  dawn::Array3i legalDimensions_;
  bool negateOffset_;

public:
  StorageResolver(ClangASTExprResolver* resolver)
      : resolver_(resolver), curDim_(-1), curArg_(-1), currentArgumentKind_(CK_Dimension),
        name_(""), offset_({{0, 0, 0}}), argumentMap_({{-1, -1, -1}}), argumentOffset_({{0, 0, 0}}),
        legalDimensions_({{0, 0, 0}}), negateOffset_(false) {}

  /// @brief Check if the typy is a storage
  /// @{
  static bool isaStorage(clang::StringRef storage) {
    if((storage.find("storage") != std::string::npos) || storage == "var")
      return true;
    return false;
  }

  static bool isaStorage(const clang::QualType& type) {
    return isaStorage(type->getAsCXXRecordDecl()->getName());
  }
  /// @}

  /// @brief Assemble FieldAccessExpr
  std::shared_ptr<dawn::sir::FieldAccessExpr>
  makeFieldAccessExpr(const dawn::SourceLocation& loc) const {
    return std::make_shared<dawn::sir::FieldAccessExpr>(name_, offset_, argumentMap_,
                                                        argumentOffset_, negateOffset_, loc);
  }

  void resolve(clang::MemberExpr* expr) {
    llvm::StringRef declName = expr->getType()->getAsCXXRecordDecl()->getName();
    std::string name = expr->getMemberNameInfo().getAsString();

    if(StorageResolver::isaStorage(expr->getType()))
      // This is a member access to a storage, set the name
      name_ = name;

    else if(declName == "dimension") {
      // This is a member access to a dimension, convert it to an index
      currentArgumentKind_ = CK_Dimension;
      curDim_ = llvm::StringSwitch<int>(name).Case("i", 0).Case("j", 1).Case("k", 2);

    } else {
      // This has to be an access to a direction or offset
      if(declName == "direction") {
        currentArgumentKind_ = CK_Direction;
      } else if(declName == "offset") {
        currentArgumentKind_ = CK_Offset;
      } else {
        resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                    Diagnostics::err_index_invalid_type)
            << expr->getType().getAsString() << name << expr->getSourceRange();
        return;
      }

      curArg_ += 1;

      // ... check if the direction (or offset) is actually an argument of the enclosing stencil
      // function and register the index in the argument list

      const auto& argDeclMap = resolver_->getParser()->getCurrentParserRecord()->CurrentArgDeclMap;
      auto it = argDeclMap.find(name);
      if(it == argDeclMap.end())
        resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                    Diagnostics::err_index_illegal_argument)
            << name << expr->getSourceRange();

      argumentMap_[curArg_] = it->second.Index;
    }
  }

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

    std::string declName = expr->getType()->getAsCXXRecordDecl()->getName();
    if(declName.find("storage") != std::string::npos) {
      legalDimensions_ = dawn::StringSwitch<dawn::Array3i>(declName)
                             .Case("storage", {{1, 1, 1}})
                             .Case("storage_i", {{1, 0, 0}})
                             .Case("storage_j", {{0, 1, 0}})
                             .Case("storage_k", {{0, 0, 1}})
                             .Case("storage_ij", {{1, 1, 0}})
                             .Case("storage_ik", {{1, 0, 1}})
                             .Case("storage_jk", {{0, 1, 1}})
                             .Case("storage_ijk", {{1, 1, 1}})
                             .Default({{0, 0, 0}});
    }
    if(declName == "var") {
      legalDimensions_ = {{1, 1, 1}};
    }
    if(expr->getOperator() == clang::OO_Call) {
      // Parse initial `u(i, j, k)` i.e `operator()(u, i, j, k)`
      resolve(dyn_cast<MemberExpr>(expr->getArg(0)));
      for(std::size_t i = 1; i < expr->getNumArgs(); ++i)
        resolve(expr->getArg(i));
    } else {
      // Parse `i` to set the `curDim_` (if `i` is a dimension) or `curArg_` (if `i` is a direction)
      resolve(expr->getArg(0));

      if(expr->getNumArgs() == 1) {
        // Parse `-off` i.e `operator-(off)` this just negates the offset
        if(expr->getOperator() == clang::OO_Minus)
          negateOffset_ = true;
      } else {
        // Parse `1` in `i+1`
        int offset = 0;
        if(IntegerLiteral* integer = dyn_cast<IntegerLiteral>(expr->getArg(1))) {
          offset = static_cast<int>(integer->getValue().signedRoundToDouble());
          offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        } else {

          // If it is not an integer literal, it may still be a constant integer expression
          Expr* arg1 = skipAllImplicitNodes(expr->getArg(1));

          DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
          clang_compat::Expr::EvalResultInt res;

          if(var && var->EvaluateAsInt(res, resolver_->getContext()->getASTContext())) {
            offset = static_cast<int>(clang_compat::Expr::getInt(res));
            offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
          } else {
            resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr->getArg(1)),
                                        Diagnostics::err_index_not_constexpr)
                << expr->getSourceRange();
            return;
          }
        }

        // Apply the offset
        if(currentArgumentKind_ == CK_Dimension) {
          offset_[curDim_] += offset;
          if((legalDimensions_[curDim_] == 0) && (offset_[curDim_] != 0)) {
            std::string dimensionOutput;
            dimensionOutput += legalDimensions_[0] ? "i " : "";
            dimensionOutput += legalDimensions_[1] ? "j " : "";
            dimensionOutput += legalDimensions_[2] ? "k " : "";
            dimensionOutput.pop_back();
            dimensionOutput += " dimensions";
            resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                        Diagnostics::err_off_with_bad_storage_dim)
                << name_ << dimensionOutput << expr->getSourceRange();
          }
        } else
          argumentOffset_[curArg_] += offset;
      }
    }
  }

private:
  void resolve(clang::CXXConstructExpr* expr) { return resolve(expr->getArg(0)); }

  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in StorageResolver");
    }
    llvm_unreachable("invalid expr");
  }
};

//===------------------------------------------------------------------------------------------===//
//     StencilFunctionArgumentResolver
//===------------------------------------------------------------------------------------------===//

/// @brief Parse an argument of a stencil function call (e.g `i+1` or simply `i` or `dir` where
/// `dir` is a direction (or offset) parameter of the current stencil function
class StencilFunctionArgumentResolver {
  ClangASTExprResolver* resolver_;

  // StencilFunArgExpr parameter
  bool isStorage_;
  int argumentIndex_;
  int direction_;
  int offset_;

public:
  StencilFunctionArgumentResolver(ClangASTExprResolver* resolver)
      : resolver_(resolver), isStorage_(false), argumentIndex_(-1), direction_(-1), offset_(0) {}

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

    // Parse `i+1` i.e `operator+(i, 1)`
    DAWN_ASSERT(expr->getNumArgs() == 2);

    // Parse `i` to set the `direction` or, in case we are in a nested stencil function call,
    // `argIndex_`
    resolve(expr->getArg(0));

    // Parse `1` in `i+1`
    if(IntegerLiteral* integer = dyn_cast<IntegerLiteral>(expr->getArg(1))) {
      offset_ = static_cast<int>(integer->getValue().signedRoundToDouble());
      offset_ *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
    } else {

      // If it is not an integer literal, it may still be a constant integer expression
      Expr* arg1 = skipAllImplicitNodes(expr->getArg(1));

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      clang_compat::Expr::EvalResultInt res;

      if(var && var->EvaluateAsInt(res, resolver_->getContext()->getASTContext())) {
        offset_ = static_cast<int>(clang_compat::Expr::getInt(res));
        offset_ *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
      } else {
        resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr->getArg(1)),
                                    Diagnostics::err_index_not_constexpr)
            << expr->getSourceRange();
      }
    }
  }

  void resolve(clang::MemberExpr* expr) {
    using namespace clang;

    // If it is a storage, it will be handled by the `StroageResolver`
    if(StorageResolver::isaStorage(expr->getType())) {
      isStorage_ = true;
      return;
    } else {
      auto name = expr->getMemberNameInfo().getAsString();

      // Handle special members `i`, `j` and `k`
      direction_ = llvm::StringSwitch<int>(name).Case("i", 0).Case("j", 1).Case("k", 2).Default(-1);
      if(direction_ != -1)
        return;

      // Check if we call it with a valid argument
      const auto& argDeclMap = resolver_->getParser()->getCurrentParserRecord()->CurrentArgDeclMap;
      auto it = argDeclMap.find(name);
      if(it == argDeclMap.end()) {
        resolver_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                    Diagnostics::err_stencilfun_invalid_argument)
            << name << expr->getSourceRange();
      }
      argumentIndex_ = it->second.Index;
    }
  }

  /// @brief Assemble StencilFunArgExpr
  ///
  /// This may return a nullptr if parsing failed. This can happen if the argument to the
  /// stencil function is a field, which will be resolved elsewhere
  std::shared_ptr<dawn::sir::StencilFunArgExpr>
  makeStencilFunArgExpr(const dawn::SourceLocation& loc) const {
    return isStorage_ ? nullptr
                      : std::make_shared<dawn::sir::StencilFunArgExpr>(direction_, offset_,
                                                                       argumentIndex_, loc);
  }

private:
  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      DAWN_ASSERT_MSG(0, "unresolved expression in StorageResolver");
    }
    llvm_unreachable("invalid expr");
  }
};

} // anonymous namespace

//===------------------------------------------------------------------------------------------===//
//     ClangASTResolver
//===------------------------------------------------------------------------------------------===//

ClangASTExprResolver::ClangASTExprResolver(GTClangContext* context, StencilParser* parser)
    : context_(context), parser_(parser),
      functionResolver_(std::make_shared<FunctionResolver>(this)) {}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::BinaryOperator* expr) {
  resetInternals();

  switch(clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())) {
  case clang::OO_Equal:
  case clang::OO_StarEqual:
  case clang::OO_SlashEqual:
  case clang::OO_PlusEqual:
  case clang::OO_MinusEqual:
    return dawn::sir::makeExprStmt(resolveAssignmentOp(expr), getSourceLocation(expr));
  default:
    return dawn::sir::makeExprStmt(
        std::make_shared<dawn::sir::BinaryOperator>(
            resolve(expr->getLHS()),
            clang::getOperatorSpelling(
                clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())),
            resolve(expr->getRHS()), getSourceLocation(expr)),
        getSourceLocation(expr));
  }
}

std::shared_ptr<dawn::sir::Stmt>
ClangASTExprResolver::resolveExpr(clang::CXXOperatorCallExpr* expr) {
  resetInternals();

  switch(expr->getOperator()) {
  case clang::OO_Equal:
  case clang::OO_StarEqual:
  case clang::OO_SlashEqual:
  case clang::OO_PlusEqual:
  case clang::OO_MinusEqual:
    return dawn::sir::makeExprStmt(resolveAssignmentOp(expr), getSourceLocation(expr));
  default:
    llvm_unreachable("invalid operator call expr");
  }
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::CXXConstructExpr* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt>
ClangASTExprResolver::resolveExpr(clang::CXXFunctionalCastExpr* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::FloatingLiteral* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::IntegerLiteral* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt>
ClangASTExprResolver::resolveExpr(clang::CXXBoolLiteralExpr* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::DeclRefExpr* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::UnaryOperator* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveExpr(clang::MemberExpr* expr) {
  resetInternals();
  return dawn::sir::makeExprStmt(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveDecl(clang::VarDecl* decl) {
  resetInternals();
  using namespace clang;

  QualType qualType = decl->getType();
  std::string varname = decl->getNameAsString();

  // If we write `avg(u);` where `avg` is a stencil function, what we actually write is `avg u;`,
  // meaning we declare a local variable of type `avg` and name `u`! At this point we have lost the
  // information that `u` was originally a storage (or a dimension). We need to recover that
  // information to construct the `StencilFunCallExpr` node.
  if(qualType.getBaseTypeIdentifier() &&
     functionResolver_->hasStencilFunction(qualType.getBaseTypeIdentifier()->getName().str())) {
    DAWN_ASSERT_MSG(0, "we currently can't parse stencil functions of the form `fun(arg1)` i.e "
                       "stencil functions with one argument and no return type!");
  }

  // Issue an error about variable shadowing i.e we declare a local variable which has the same name
  // as a member of the surrounding stencil or stencil-function
  const auto& argDeclMap = parser_->getCurrentParserRecord()->CurrentArgDeclMap;
  auto it = parser_->getCurrentParserRecord()->CurrentArgDeclMap.find(varname);
  if(it != argDeclMap.end()) {
    reportDiagnostic(clang_compat::getBeginLoc(*decl), Diagnostics::err_do_method_var_shadowing)
        << varname
        << (parser_->getCurrentParserRecord()->CurrentKind == StencilParser::SK_Stencil
                ? "stencil"
                : "stencil function");
    reportDiagnostic(it->second.Decl->getLocation(), Diagnostics::note_previous_definition);
  }

  // Extract builtin type (if any)
  const Type* type = qualType->isArrayType() ? qualType->getArrayElementTypeNoTypeQual()
                                             : qualType.getTypePtrOrNull();
  DAWN_ASSERT(type);

  dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Invalid;
  if(type->isBuiltinType()) {
    if(type->isBooleanType()) // bool
      builtinType = dawn::BuiltinTypeID::Boolean;
    else if(type->isIntegerType()) // int
      builtinType = dawn::BuiltinTypeID::Integer;
    else // int, float, double...
      builtinType = dawn::BuiltinTypeID::Float;
  } else {
    reportDiagnostic(clang_compat::getBeginLoc(*decl),
                     Diagnostics::err_do_method_invalid_type_of_local_var)
        << qualType.getAsString() << varname;
    return nullptr;
  }

  // Extract qualifiers (if any)
  dawn::CVQualifier cvQualifier = dawn::CVQualifier::Invalid;
  if(qualType.hasQualifiers()) {
    if(qualType.isConstQualified())
      cvQualifier |= dawn::CVQualifier::Const;
    if(qualType.isVolatileQualified())
      cvQualifier |= dawn::CVQualifier::Volatile;
  }

  // Assemble the type
  dawn::Type DAWNType;
  if(builtinType != dawn::BuiltinTypeID::Invalid)
    DAWNType = dawn::Type(builtinType, cvQualifier);
  else
    DAWNType = dawn::Type(qualType.getAsString(), cvQualifier);

  // Set dimension
  int dimension = 0;
  if(qualType->isArrayType()) {
    if(!qualType->isConstantArrayType()) {
      reportDiagnostic(clang_compat::getBeginLoc(*decl),
                       Diagnostics::err_do_method_non_const_array_type)
          << qualType.getAsString() << varname;
      return nullptr;
    }
    const ConstantArrayType* constArrayType = dyn_cast<ConstantArrayType>(qualType);
    dimension = static_cast<int>(constArrayType->getSize().roundToDouble());
  }

  // Resolve initializer list
  std::vector<std::shared_ptr<dawn::sir::Expr>> initList;
  if(decl->hasInit()) {
    if(InitListExpr* varInitList = dyn_cast<InitListExpr>(decl->getInit())) {
      for(Expr* init : varInitList->inits())
        initList.push_back(resolve(init));
    } else
      initList.push_back(resolve(decl->getInit()));
  }

  return dawn::sir::makeVarDeclStmt(DAWNType, varname, dimension, "=", std::move(initList),
                                    getSourceLocation(decl));
}

std::shared_ptr<dawn::sir::Stmt> ClangASTExprResolver::resolveStmt(clang::ReturnStmt* stmt) {
  resetInternals();
  return dawn::sir::makeReturnStmt(resolve(stmt->getRetValue()), getSourceLocation(stmt));
}

dawn::SourceLocation ClangASTExprResolver::getSourceLocation(clang::Stmt* stmt) const {
  clang::PresumedLoc ploc =
      context_->getSourceManager().getPresumedLoc(clang_compat::getBeginLoc(*stmt));
  return dawn::SourceLocation(ploc.getLine(), ploc.getColumn());
}

dawn::SourceLocation ClangASTExprResolver::getSourceLocation(clang::Decl* decl) const {
  clang::PresumedLoc ploc =
      context_->getSourceManager().getPresumedLoc(clang_compat::getBeginLoc(*decl));
  return dawn::SourceLocation(ploc.getLine(), ploc.getColumn());
}

//===------------------------------------------------------------------------------------------===//
//     Internal expression resolver

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::Expr* expr) {
  using namespace clang;
  // ignore implicit nodes
  expr = skipAllImplicitNodes(expr);

  if(ArraySubscriptExpr* e = dyn_cast<ArraySubscriptExpr>(expr))
    return resolve(e);
  else if(BinaryOperator* e = dyn_cast<BinaryOperator>(expr))
    return resolve(e);
  else if(CStyleCastExpr* e = dyn_cast<CStyleCastExpr>(expr))
    return resolve(e);
  else if(CXXBoolLiteralExpr* e = dyn_cast<CXXBoolLiteralExpr>(expr))
    return resolve(e);
  else if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
    return resolve(e);
  else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
    return resolve(e);
  else if(CXXMemberCallExpr* e = dyn_cast<CXXMemberCallExpr>(expr))
    return resolve(e);
  else if(CXXFunctionalCastExpr* e = dyn_cast<CXXFunctionalCastExpr>(expr))
    return resolve(e);
  else if(ConditionalOperator* e = dyn_cast<ConditionalOperator>(expr))
    return resolve(e);
  else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
    return resolve(e);
  else if(FloatingLiteral* e = dyn_cast<FloatingLiteral>(expr))
    return resolve(e);
  else if(IntegerLiteral* e = dyn_cast<IntegerLiteral>(expr))
    return resolve(e);
  else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
    return resolve(e);
  else if(ParenExpr* e = dyn_cast<ParenExpr>(expr))
    return resolve(e);
  else if(UnaryOperator* e = dyn_cast<UnaryOperator>(expr))
    return resolve(e);

  // These should be handled last as we want to select the derived classes first  (e.g a
  // `CXXMemberCallExpr` is a `CallExpr`)
  else if(CallExpr* e = dyn_cast<CallExpr>(expr))
    return resolve(e);

  else {
    expr->dumpColor();
    DAWN_ASSERT_MSG(0, "unresolved expression");
  }
  llvm_unreachable("invalid expr");
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::ArraySubscriptExpr* expr) {
  // Resolve the variable
  auto DAWNExpr = resolve(expr->getBase());

  auto DAWNVarAccessExpr = dawn::dyn_cast<dawn::sir::VarAccessExpr>(DAWNExpr.get());
  DAWN_ASSERT(DAWNVarAccessExpr);

  // Resolve the index
  DAWNVarAccessExpr->setIndex(resolve(expr->getIdx()));
  return DAWNExpr;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::BinaryOperator* expr) {
  if(functionResolver_->isActive())
    functionResolver_->skipNextArguments(2);

  auto binary = std::make_shared<dawn::sir::BinaryOperator>(
      resolve(expr->getLHS()),
      clang::getOperatorSpelling(clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())),
      resolve(expr->getRHS()), getSourceLocation(expr));

  if(functionResolver_->isActive())
    functionResolver_->addArgument(expr, binary);

  if(functionResolver_->isActiveOnStencilFunction())
    reportDiagnostic(clang_compat::getBeginLoc(*expr),
                     Diagnostics::err_stencilfun_expression_in_arg_list)
        << expr->getSourceRange();

  return binary;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CallExpr* expr) {
  bool isNestedFunction = functionResolver_->isActive();
  functionResolver_->push(expr);

  // Resolve the arguments
  for(clang::Expr* arg : expr->arguments())
    resolve(arg);

  auto function = functionResolver_->pop();
  if(isNestedFunction)
    functionResolver_->addArgument(expr, function);
  return function;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CStyleCastExpr* expr) {
  currentCStyleCastExpr_ = expr;
  return resolve(expr->getSubExpr());
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CXXBoolLiteralExpr* expr) {
  auto booleanLiteral = std::make_shared<dawn::sir::LiteralAccessExpr>(
      expr->getValue() ? "true" : "false", resolveBuiltinType(expr), getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, booleanLiteral);

  return booleanLiteral;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CXXOperatorCallExpr* expr) {
  if(expr->getOperator() == clang::OO_Call) {

    // This should be a call to a storage with offset (e.g `u(i, j, k)`)
    StorageResolver resolver(this);
    resolver.resolve(expr);
    auto field = resolver.makeFieldAccessExpr(getSourceLocation(expr));

    if(functionResolver_->isActive())
      // We are inside an argument-list of a stencil function, append the arguments
      functionResolver_->addArgument(expr, field);

    return field;

  } else if(expr->getOperator() == clang::OO_Plus || expr->getOperator() == clang::OO_Minus) {
    // This is an argument to a stencil function call `i+1` or `dir+1`
    DAWN_ASSERT(functionResolver_->isActive());

    StencilFunctionArgumentResolver resolver(this);
    resolver.resolve(expr);
    return functionResolver_->addArgument(expr,
                                          resolver.makeStencilFunArgExpr(getSourceLocation(expr)));
  }
  llvm_unreachable("invalid operator");
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CXXConstructExpr* expr) {
  if(StorageResolver::isaStorage(getClassNameFromConstructExpr(expr)) && expr->getNumArgs() == 1) {
    // This is an access to a storage e.g `u = ...`
    return resolve(expr->getArg(0));

  } else if(functionResolver_->checkAndSetReferenced()) {

    // We are currently parsing a non-nested stencil function and this is the first time we
    // encounter another `CXXConstructExpr`. There are two options here:
    //
    // 1) We got referenced by a `CXXTemporaryObjectExpr` or a `CXXConstructExpr` (this happens
    //    sometimes -- quirk in C++). We will just ignore the expr and parse the argument list.
    //    Note that appareantly we can only get refernced if we were *not* constructed with a
    //    `CXXTemporaryObjectExpr`. Yes.. this seems very hacky here..
    // 2) We are starting to parse a nested stencil function and we would need to hit the next
    //    branch.
    //
    for(clang::Expr* arg : expr->arguments())
      resolve(arg);

    return nullptr;

  } else {
    // Resolve stencil function
    bool isNestedFunction = functionResolver_->isActive();

    // This is a stencil function call (e.g `avg(u, v)`), make sure the stencil function exists
    if(!functionResolver_->push(expr))
      return nullptr;

    // Resolve the arguments
    for(clang::Expr* arg : expr->arguments())
      resolve(arg);

    // Check number of arguments matches
    functionResolver_->checkNumOfArgumentsMatch();

    // If we parse a nested stencil function, we need to register it as an argument
    std::shared_ptr<dawn::sir::StencilFunCallExpr> stencilFunctionFun =
        std::static_pointer_cast<dawn::sir::StencilFunCallExpr>(functionResolver_->pop());
    if(isNestedFunction)
      functionResolver_->addArgument(expr, stencilFunctionFun);

    return stencilFunctionFun;
  }
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CXXMemberCallExpr* expr) {
  std::string methodName = expr->getMethodDecl()->getNameInfo().getName().getAsString();
  if(methodName == "operator double") {
    // This covers 2 cases:
    //
    //  Case 1: This is the result of a conversion: `operator double()`
    //          It is triggered for example in the following expr: `1.0 + v`
    //          where v is converted into int. We need to get the callee
    //
    //  Case 2: It is a stencil function call like `v = delta(u)`
    //          where the return statement is casted via `operator double()`
    clang::MemberExpr* e = clang::dyn_cast<clang::MemberExpr>(expr->getCallee());
    DAWN_ASSERT(e);
    DAWN_ASSERT(e->getBase());
    return resolve(e->getBase());
  }
  llvm_unreachable("invalid CXXMemberCallExpr");
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::CXXFunctionalCastExpr* expr) {
  return resolve(expr->getSubExpr());
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::ConditionalOperator* expr) {
  return std::make_shared<dawn::sir::TernaryOperator>(
      resolve(expr->getCond()), resolve(expr->getLHS()), resolve(expr->getRHS()));
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::DeclRefExpr* expr) {
  // Check we don't reference non-local variables (this will not end well in CUDA code)
  if(clang::VarDecl* varDecl = clang::dyn_cast<clang::VarDecl>(expr->getDecl())) {
    if(!varDecl->isLocalVarDecl()) {
      reportDiagnostic(expr->getLocation(), Diagnostics::err_do_method_non_local_var)
          << varDecl->getName() << expr->getSourceRange();
      reportDiagnostic(varDecl->getLocation(), Diagnostics::note_declaration);
      return nullptr;
    }
  }

  // Access to a locally declared variable
  auto var = std::make_shared<dawn::sir::VarAccessExpr>(expr->getNameInfo().getAsString(), nullptr,
                                                        getSourceLocation(expr));

  if(functionResolver_->isActive())
    return functionResolver_->addArgument(expr, var);
  return var;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::FloatingLiteral* expr) {
  llvm::SmallVector<char, 10> valueVec;
  expr->getValue().toString(valueVec);
  auto floatLiteral = std::make_shared<dawn::sir::LiteralAccessExpr>(
      std::string(valueVec.data(), valueVec.size()), resolveBuiltinType(expr),
      getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, floatLiteral);

  return floatLiteral;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::IntegerLiteral* expr) {
  auto integerLiteral = std::make_shared<dawn::sir::LiteralAccessExpr>(
      expr->getValue().toString(10, true), resolveBuiltinType(expr), getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, integerLiteral);

  return integerLiteral;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::MemberExpr* expr) {
  llvm::StringRef baseIdentifier = expr->getBase()->getType().getBaseTypeIdentifier()->getName();

  if(baseIdentifier == "globals") {
    // Access to a global variable
    auto var = std::make_shared<dawn::sir::VarAccessExpr>(expr->getMemberNameInfo().getAsString(),
                                                          nullptr, getSourceLocation(expr));
    var->setIsExternal(true);
    DAWN_ASSERT_MSG(parser_->isGlobalVariable(var->getName()),
                    "access to unregistered global variable");

    if(functionResolver_->isActive())
      // We are inside an argument-list of a c++ function, append the arguments
      return functionResolver_->addArgument(expr, var);
    return var;
  }

  // Handle special arguments to stencil functions which can be directions, offsets or dimensions
  if(functionResolver_->isActive()) {

    StencilFunctionArgumentResolver resolver(this);
    resolver.resolve(expr);
    auto arg = resolver.makeStencilFunArgExpr(getSourceLocation(expr));

    // arg is a field, we handle it below (i.e arg is a dimension or offset in this case)
    if(arg)
      return functionResolver_->addArgument(expr, arg);
  }

  // This should be a call to a storage without any offsets (e.g `u`)
  StorageResolver resolver(this);
  resolver.resolve(expr);
  auto field = resolver.makeFieldAccessExpr(getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a stencil function, append the arguments
    return functionResolver_->addArgument(expr, field);

  return field;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::ParenExpr* expr) {
  return resolve(expr->getSubExpr());
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolve(clang::UnaryOperator* expr) {
  if(functionResolver_->isActive())
    functionResolver_->skipNextArguments(1);

  auto unary = std::make_shared<dawn::sir::UnaryOperator>(
      resolve(expr->getSubExpr()),
      clang::getOperatorSpelling(clang::UnaryOperator::getOverloadedOperator(expr->getOpcode())),
      getSourceLocation(expr));

  if(functionResolver_->isActive())
    functionResolver_->addArgument(expr, unary);

  if(functionResolver_->isActiveOnStencilFunction())
    reportDiagnostic(clang_compat::getBeginLoc(*expr),
                     Diagnostics::err_stencilfun_expression_in_arg_list)
        << expr->getSourceRange();

  return unary;
}

std::shared_ptr<dawn::sir::Expr> ClangASTExprResolver::resolveAssignmentOp(clang::Expr* expr) {
  if(clang::CXXOperatorCallExpr* e = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    DAWN_ASSERT(e->getNumArgs() == 2);
    return std::make_shared<dawn::sir::AssignmentExpr>(resolve(e->getArg(0)), resolve(e->getArg(1)),
                                                       clang::getOperatorSpelling(e->getOperator()),
                                                       getSourceLocation(e));
  } else if(clang::BinaryOperator* e = clang::dyn_cast<clang::BinaryOperator>(expr)) {
    return std::make_shared<dawn::sir::AssignmentExpr>(
        resolve(e->getLHS()), resolve(e->getRHS()),
        clang::getOperatorSpelling(clang::BinaryOperator::getOverloadedOperator(e->getOpcode())),
        getSourceLocation(e));
  }
  llvm_unreachable("invalid assignment expr");
}

dawn::BuiltinTypeID ClangASTExprResolver::resolveBuiltinType(clang::Expr* expr) {
  using namespace clang;
  // ignore implicit nodes
  expr = skipAllImplicitNodes(expr);

  // Resolve C-Style casting
  if(currentCStyleCastExpr_) {
    QualType type = currentCStyleCastExpr_->getType();
    DAWN_ASSERT(!type.isNull());

    // Reset (we consumed the casting)
    currentCStyleCastExpr_ = nullptr;

    if(type->isBooleanType()) // bool
      return dawn::BuiltinTypeID::Boolean;
    else if(type->isIntegerType()) // int
      return dawn::BuiltinTypeID::Integer;
    else if(type->isArithmeticType()) // int, float, double...
      return dawn::BuiltinTypeID::Float;
    else
      DAWN_LOG(WARNING) << "ignoring C-Style cast to " << type.getAsString();
  }

  // Resolve the builtin type and apply potential casting
  if(isa<IntegerLiteral>(expr)) {
    return dawn::BuiltinTypeID::Integer;
  } else if(isa<FloatingLiteral>(expr)) {
    return dawn::BuiltinTypeID::Float;
  } else if(isa<CXXBoolLiteralExpr>(expr)) {
    return dawn::BuiltinTypeID::Boolean;
  }
  llvm_unreachable("invalid expr for builtin type");
}

void ClangASTExprResolver::resetInternals() {
  currentCStyleCastExpr_ = nullptr;
  functionResolver_->reset();
}

clang::DiagnosticBuilder ClangASTExprResolver::reportDiagnostic(clang::SourceLocation loc,
                                                                Diagnostics::DiagKind kind) {
  return context_->getDiagnostics().report(loc, kind);
}

} // namespace gtclang
