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

#include "gtclang/Frontend/ClangASTExprResolver.h"
#include "gsl/SIR/AST.h"
#include "gsl/Support/Assert.h"
#include "gsl/Support/Casting.h"
#include "gsl/Support/StringUtil.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/StencilParser.h"
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
    std::shared_ptr<gsl::sir::StencilFunction> SIRStencilFunction = nullptr;

    /// AST node of the C++ or stencil function i.e the result of the parsing
    std::shared_ptr<gsl::FunCallExpr> GSLFunCallExpr = nullptr;

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
    scope_.push(llvm::make_unique<Scope>());
    scope_.top()->Kind = FK_StencilFunction;

    std::string callee = expr->getConstructor()->getNameAsString();
    scope_.top()->GSLFunCallExpr =
        std::make_shared<gsl::StencilFunCallExpr>(callee, resolver_->getSourceLocation(expr));
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
    scope_.push(llvm::make_unique<Scope>());
    scope_.top()->Kind = FK_CXXFunction;

    GSL_ASSERT_MSG(expr->getDirectCallee(), "only C-function calls are supported");
    scope_.top()->GSLFunCallExpr = std::make_shared<gsl::FunCallExpr>(
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
  std::shared_ptr<gsl::Expr> addArgument(clang::Expr* expr, const std::shared_ptr<gsl::Expr>& arg) {
    GSL_ASSERT(isActive() && scope_.top()->GSLFunCallExpr);

    if(scope_.top()->ArgumentsToSkip != 0) {
      scope_.top()->ArgumentsToSkip--;
      return arg;
    }

    scope_.top()->GSLFunCallExpr->getArguments().emplace_back(arg);

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
  std::shared_ptr<gsl::FunCallExpr> pop() {
    auto expr = scope_.top()->GSLFunCallExpr;
    scope_.pop();
    return expr;
  }

  /// @brief Check if number of arguments match
  void checkNumOfArgumentsMatch() {
    GSL_ASSERT(isActive() && scope_.top()->GSLFunCallExpr);
    GSL_ASSERT(getCurrentKind() == FK_StencilFunction);

    std::size_t requiredArgs = scope_.top()->SIRStencilFunction->Args.size();
    std::size_t parsedArgs = scope_.top()->GSLFunCallExpr->getArguments().size();

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
    GSL_ASSERT(isActive());
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
          << ClangCXXConstructExpr->getConstructor()->getNameAsString()
          << ClangCXXConstructExpr->getSourceRange();
      scope_.top()->DiagnosticIssued = true;
      return true;
    }
    return false;
  }

  /// @brief Check if types of arguments match
  void checkTypeOfArgumentMatches(clang::Expr* expr, gsl::Expr* parsedArg) {
    int curIndex = scope_.top()->GSLFunCallExpr->getArguments().size() - 1;

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
         << "' for " << gsl::decimalToOrdinal(curIndex + 1) << " argument";
      reportNote(ss.str());
    }
  }

  /// @brief Convert an SIR stencil function argument to the matching index in `TypeStr`
  static ArgumentIndex getIndexFromSIRArg(gsl::sir::StencilFunctionArg* arg) {
    if(gsl::isa<gsl::sir::Field>(arg))
      return AK_Field;
    else if(gsl::isa<gsl::sir::Direction>(arg))
      return AK_Direction;
    else // == gsl::sir::Offset
      return AK_Offset;
  };

  /// @brief Convert an AST stencil function argument to the matching index in `TypeStr`
  static ArgumentIndex getIndexFromASTArg(gsl::Expr* expr) {
    if(gsl::isa<gsl::FieldAccessExpr>(expr) || gsl::isa<gsl::StencilFunCallExpr>(expr))
      return AK_Field;

    gsl::StencilFunArgExpr* arg = gsl::dyn_cast<gsl::StencilFunArgExpr>(expr);
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
  gsl::Array3i offset_;
  gsl::Array3i argumentMap_;
  gsl::Array3i argumentOffset_;
  bool negateOffset_;

public:
  StorageResolver(ClangASTExprResolver* resolver)
      : resolver_(resolver), curDim_(-1), curArg_(-1), currentArgumentKind_(CK_Dimension),
        name_(""), offset_({{0, 0, 0}}), argumentMap_({{-1, -1, -1}}), argumentOffset_({{0, 0, 0}}),
        negateOffset_(false) {}

  /// @brief Check if the typy is a storage
  /// @{
  static bool isaStorage(clang::StringRef storage) {
    if(storage == "storage" || storage == "temporary_storage")
      return true;
    return false;
  }

  static bool isaStorage(const clang::QualType& type) {
    return isaStorage(type->getAsCXXRecordDecl()->getName());
  }
  /// @}

  /// @brief Assemble FieldAccessExpr
  std::shared_ptr<gsl::FieldAccessExpr> makeFieldAccessExpr(const gsl::SourceLocation& loc) const {
    return std::make_shared<gsl::FieldAccessExpr>(name_, offset_, argumentMap_, argumentOffset_,
                                                  negateOffset_, loc);
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
        resolver_->reportDiagnostic(expr->getLocStart(), Diagnostics::err_index_invalid_type)
            << expr->getType().getAsString() << name << expr->getSourceRange();
        return;
      }

      curArg_ += 1;

      // ... check if the direction (or offset) is actually an argument of the enclosing stencil
      // function and register the index in the argument list

      const auto& argDeclMap = resolver_->getParser()->getCurrentParserRecord()->CurrentArgDeclMap;
      auto it = argDeclMap.find(name);
      if(it == argDeclMap.end())
        resolver_->reportDiagnostic(expr->getLocStart(), Diagnostics::err_index_illegal_argument)
            << name << expr->getSourceRange();

      argumentMap_[curArg_] = it->second.Index;
    }
  }

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

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
          Expr* arg1 = expr->getArg(1);
          if(ImplicitCastExpr* castExpr = dyn_cast<ImplicitCastExpr>(arg1))
            arg1 = castExpr->getSubExpr();

          DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
          llvm::APSInt res;

          if(var && var->EvaluateAsInt(res, resolver_->getContext()->getASTContext())) {
            offset = static_cast<int>(res.getExtValue());
            offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
          } else {
            resolver_->reportDiagnostic(expr->getArg(1)->getLocStart(),
                                        Diagnostics::err_index_not_constexpr)
                << expr->getSourceRange();
            return;
          }
        }

        // Apply the offset
        if(currentArgumentKind_ == CK_Dimension)
          offset_[curDim_] += offset;
        else
          argumentOffset_[curArg_] += offset;
      }
    }
  }

private:
  void resolve(clang::CXXConstructExpr* expr) { return resolve(expr->getArg(0)); }

  void resolve(clang::ImplicitCastExpr* expr) { return resolve(expr->getSubExpr()); }

  void resolve(clang::MaterializeTemporaryExpr* expr) { return resolve(expr->GetTemporaryExpr()); }

  void resolve(clang::Expr* expr) {
    using namespace clang;

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(ImplicitCastExpr* e = dyn_cast<ImplicitCastExpr>(expr))
      return resolve(e);
    else if(MaterializeTemporaryExpr* e = dyn_cast<MaterializeTemporaryExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      GSL_ASSERT_MSG(0, "unresolved expression in StorageResolver");
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
    GSL_ASSERT(expr->getNumArgs() == 2);

    // Parse `i` to set the `direction` or, in case we are in a nested stencil function call,
    // `argIndex_`
    resolve(expr->getArg(0));

    // Parse `1` in `i+1`
    if(IntegerLiteral* integer = dyn_cast<IntegerLiteral>(expr->getArg(1))) {
      offset_ = static_cast<int>(integer->getValue().signedRoundToDouble());
      offset_ *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
    } else {

      // If it is not an integer literal, it may still be a constant integer expression
      Expr* arg1 = expr->getArg(1);
      if(ImplicitCastExpr* castExpr = dyn_cast<ImplicitCastExpr>(arg1))
        arg1 = castExpr->getSubExpr();

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      llvm::APSInt res;

      if(var && var->EvaluateAsInt(res, resolver_->getContext()->getASTContext())) {
        offset_ = static_cast<int>(res.getExtValue());
        offset_ *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
      } else {
        resolver_->reportDiagnostic(expr->getArg(1)->getLocStart(),
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
        resolver_->reportDiagnostic(expr->getLocStart(),
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
  std::shared_ptr<gsl::StencilFunArgExpr>
  makeStencilFunArgExpr(const gsl::SourceLocation& loc) const {
    return isStorage_ ? nullptr : std::make_shared<gsl::StencilFunArgExpr>(direction_, offset_,
                                                                           argumentIndex_, loc);
  }

private:
  void resolve(clang::Expr* expr) {
    using namespace clang;

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      GSL_ASSERT_MSG(0, "unresolved expression in StorageResolver");
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

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::BinaryOperator* expr) {
  resetInternals();

  switch(clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())) {
  case clang::OO_Equal:
  case clang::OO_StarEqual:
  case clang::OO_SlashEqual:
  case clang::OO_PlusEqual:
  case clang::OO_MinusEqual:
    return std::make_shared<gsl::ExprStmt>(resolveAssignmentOp(expr), getSourceLocation(expr));
  default:
    return std::make_shared<gsl::ExprStmt>(
        std::make_shared<gsl::BinaryOperator>(
            resolve(expr->getLHS()),
            clang::getOperatorSpelling(
                clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())),
            resolve(expr->getRHS()), getSourceLocation(expr)),
        getSourceLocation(expr));
  }
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::CXXOperatorCallExpr* expr) {
  resetInternals();

  switch(expr->getOperator()) {
  case clang::OO_Equal:
  case clang::OO_StarEqual:
  case clang::OO_SlashEqual:
  case clang::OO_PlusEqual:
  case clang::OO_MinusEqual:
    return std::make_shared<gsl::ExprStmt>(resolveAssignmentOp(expr), getSourceLocation(expr));
  default:
    llvm_unreachable("invalid operator call expr");
  }
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::CXXConstructExpr* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::CXXFunctionalCastExpr* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::FloatingLiteral* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::IntegerLiteral* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::CXXBoolLiteralExpr* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::DeclRefExpr* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::UnaryOperator* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveExpr(clang::MemberExpr* expr) {
  resetInternals();
  return std::make_shared<gsl::ExprStmt>(resolve(expr), getSourceLocation(expr));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveDecl(clang::VarDecl* decl) {
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
    GSL_ASSERT_MSG(0, "we currently can't parse stencil functions of the form `fun(arg1)` i.e "
                      "stencil functions with one argument and no return type!");
  }

  // Issue an error about variable shadowing i.e we declare a local variable which has the same name
  // as a member of the surrounding stencil or stencil-function
  const auto& argDeclMap = parser_->getCurrentParserRecord()->CurrentArgDeclMap;
  auto it = parser_->getCurrentParserRecord()->CurrentArgDeclMap.find(varname);
  if(it != argDeclMap.end()) {
    reportDiagnostic(decl->getLocStart(), Diagnostics::err_do_method_var_shadowing)
        << varname << (parser_->getCurrentParserRecord()->CurrentKind == StencilParser::SK_Stencil
                           ? "stencil"
                           : "stencil function");
    reportDiagnostic(it->second.Decl->getLocation(), Diagnostics::note_previous_definition);
  }

  // Extract builtin type (if any)
  const Type* type = qualType->isArrayType() ? qualType->getArrayElementTypeNoTypeQual()
                                             : qualType.getTypePtrOrNull();
  GSL_ASSERT(type);

  gsl::BuiltinTypeID builtinType = gsl::BuiltinTypeID::None;
  if(type->isBuiltinType()) {
    if(type->isBooleanType()) // bool
      builtinType = gsl::BuiltinTypeID::Boolean;
    else if(type->isIntegerType()) // int
      builtinType = gsl::BuiltinTypeID::Integer;
    else // int, float, double...
      builtinType = gsl::BuiltinTypeID::Float;
  } else {
    reportDiagnostic(decl->getLocStart(), Diagnostics::err_do_method_invalid_type_of_local_var)
        << qualType.getAsString() << varname;
    return nullptr;
  }

  // Extract qualifiers (if any)
  gsl::CVQualifier cvQualifier = gsl::CVQualifier::None;
  if(qualType.hasQualifiers()) {
    if(qualType.isConstQualified())
      cvQualifier |= gsl::CVQualifier::Const;
    if(qualType.isVolatileQualified())
      cvQualifier |= gsl::CVQualifier::Volatile;
  }

  // Assemble the type
  gsl::Type GSLType;
  if(builtinType != gsl::BuiltinTypeID::None)
    GSLType = gsl::Type(builtinType, cvQualifier);
  else
    GSLType = gsl::Type(qualType.getAsString(), cvQualifier);

  // Set dimension
  int dimension = 0;
  if(qualType->isArrayType()) {
    if(!qualType->isConstantArrayType()) {
      reportDiagnostic(decl->getLocStart(), Diagnostics::err_do_method_non_const_array_type)
          << qualType.getAsString() << varname;
      return nullptr;
    }
    const ConstantArrayType* constArrayType = dyn_cast<ConstantArrayType>(qualType);
    dimension = static_cast<int>(constArrayType->getSize().roundToDouble());
  }

  // Resolve initializer list
  std::vector<std::shared_ptr<gsl::Expr>> initList;
  if(decl->hasInit()) {
    if(InitListExpr* varInitList = dyn_cast<InitListExpr>(decl->getInit())) {
      for(Expr* init : varInitList->inits())
        initList.push_back(resolve(init));
    } else
      initList.push_back(resolve(decl->getInit()));
  }

  return std::make_shared<gsl::VarDeclStmt>(GSLType, varname, dimension, "=", std::move(initList),
                                            getSourceLocation(decl));
}

std::shared_ptr<gsl::Stmt> ClangASTExprResolver::resolveStmt(clang::ReturnStmt* stmt) {
  resetInternals();
  return std::make_shared<gsl::ReturnStmt>(resolve(stmt->getRetValue()), getSourceLocation(stmt));
}

gsl::SourceLocation ClangASTExprResolver::getSourceLocation(clang::Stmt* stmt) const {
  clang::PresumedLoc ploc = context_->getSourceManager().getPresumedLoc(stmt->getLocStart());
  return gsl::SourceLocation(ploc.getLine(), ploc.getColumn());
}

gsl::SourceLocation ClangASTExprResolver::getSourceLocation(clang::Decl* decl) const {
  clang::PresumedLoc ploc = context_->getSourceManager().getPresumedLoc(decl->getLocStart());
  return gsl::SourceLocation(ploc.getLine(), ploc.getColumn());
}

//===------------------------------------------------------------------------------------------===//
//     Internal expression resolver

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::Expr* expr) {
  using namespace clang;

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
  else if(ImplicitCastExpr* e = dyn_cast<ImplicitCastExpr>(expr))
    return resolve(e);
  else if(IntegerLiteral* e = dyn_cast<IntegerLiteral>(expr))
    return resolve(e);
  else if(MaterializeTemporaryExpr* e = dyn_cast<MaterializeTemporaryExpr>(expr))
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
    GSL_ASSERT_MSG(0, "unresolved expression");
  }
  llvm_unreachable("invalid expr");
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::ArraySubscriptExpr* expr) {
  // Resolve the variable
  auto GSLExpr = resolve(expr->getBase());

  auto GSLVarAccessExpr = gsl::dyn_cast<gsl::VarAccessExpr>(GSLExpr.get());
  GSL_ASSERT(GSLVarAccessExpr);

  // Resolve the index
  GSLVarAccessExpr->setIndex(resolve(expr->getIdx()));
  return GSLExpr;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::BinaryOperator* expr) {
  if(functionResolver_->isActive())
    functionResolver_->skipNextArguments(2);

  auto binary = std::make_shared<gsl::BinaryOperator>(
      resolve(expr->getLHS()),
      clang::getOperatorSpelling(clang::BinaryOperator::getOverloadedOperator(expr->getOpcode())),
      resolve(expr->getRHS()), getSourceLocation(expr));

  if(functionResolver_->isActive())
    functionResolver_->addArgument(expr, binary);

  if(functionResolver_->isActiveOnStencilFunction())
    reportDiagnostic(expr->getLocStart(), Diagnostics::err_stencilfun_expression_in_arg_list)
        << expr->getSourceRange();

  return binary;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CallExpr* expr) {
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

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CStyleCastExpr* expr) {
  currentCStyleCastExpr_ = expr;
  return resolve(expr->getSubExpr());
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CXXBoolLiteralExpr* expr) {
  auto booleanLiteral = std::make_shared<gsl::LiteralAccessExpr>(
      expr->getValue() ? "true" : "false", resolveBuiltinType(expr), getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, booleanLiteral);

  return booleanLiteral;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CXXOperatorCallExpr* expr) {
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
    GSL_ASSERT(functionResolver_->isActive());

    StencilFunctionArgumentResolver resolver(this);
    resolver.resolve(expr);
    return functionResolver_->addArgument(expr,
                                          resolver.makeStencilFunArgExpr(getSourceLocation(expr)));
  }
  llvm_unreachable("invalid operator");
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CXXConstructExpr* expr) {
  if(StorageResolver::isaStorage(expr->getConstructor()->getNameAsString()) &&
     expr->getNumArgs() == 1) {
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
      std::shared_ptr<gsl::StencilFunCallExpr> stencilFunctionFun =
          std::static_pointer_cast<gsl::StencilFunCallExpr>(functionResolver_->pop());
      if(isNestedFunction)
        functionResolver_->addArgument(expr, stencilFunctionFun);

      return stencilFunctionFun;
  }
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CXXMemberCallExpr* expr) {
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
    GSL_ASSERT(e);
    GSL_ASSERT(e->getBase());
    return resolve(e->getBase());
  }
  llvm_unreachable("invalid CXXMemberCallExpr");
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::CXXFunctionalCastExpr* expr) {
  return resolve(expr->getSubExpr());
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::ConditionalOperator* expr) {
  return std::make_shared<gsl::TernaryOperator>(resolve(expr->getCond()), resolve(expr->getLHS()),
                                                resolve(expr->getRHS()));
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::DeclRefExpr* expr) {
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
  auto var = std::make_shared<gsl::VarAccessExpr>(expr->getNameInfo().getAsString(), nullptr,
                                                  getSourceLocation(expr));

  if(functionResolver_->isActive())
    return functionResolver_->addArgument(expr, var);
  return var;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::FloatingLiteral* expr) {
  llvm::SmallVector<char, 10> valueVec;
  expr->getValue().toString(valueVec);
  auto floatLiteral =
      std::make_shared<gsl::LiteralAccessExpr>(std::string(valueVec.data(), valueVec.size()),
                                               resolveBuiltinType(expr), getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, floatLiteral);

  return floatLiteral;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::ImplicitCastExpr* expr) {
  return resolve(expr->getSubExpr());
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::IntegerLiteral* expr) {
  auto integerLiteral = std::make_shared<gsl::LiteralAccessExpr>(
      expr->getValue().toString(10, true), resolveBuiltinType(expr), getSourceLocation(expr));

  if(functionResolver_->isActive())
    // We are inside an argument-list of a function, append the arguments
    return functionResolver_->addArgument(expr, integerLiteral);

  return integerLiteral;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::MaterializeTemporaryExpr* expr) {
  return resolve(expr->GetTemporaryExpr());
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::MemberExpr* expr) {
  llvm::StringRef baseIdentifier = expr->getBase()->getType().getBaseTypeIdentifier()->getName();

  if(baseIdentifier == "globals") {
    // Access to a global variable
    auto var = std::make_shared<gsl::VarAccessExpr>(expr->getMemberNameInfo().getAsString(),
                                                    nullptr, getSourceLocation(expr));
    var->setIsExternal(true);
    GSL_ASSERT_MSG(parser_->isGlobalVariable(var->getName()),
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

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::ParenExpr* expr) {
  return resolve(expr->getSubExpr());
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolve(clang::UnaryOperator* expr) {
  if(functionResolver_->isActive())
    functionResolver_->skipNextArguments(1);

  auto unary = std::make_shared<gsl::UnaryOperator>(
      resolve(expr->getSubExpr()),
      clang::getOperatorSpelling(clang::UnaryOperator::getOverloadedOperator(expr->getOpcode())),
      getSourceLocation(expr));

  if(functionResolver_->isActive())
    functionResolver_->addArgument(expr, unary);

  if(functionResolver_->isActiveOnStencilFunction())
    reportDiagnostic(expr->getLocStart(), Diagnostics::err_stencilfun_expression_in_arg_list)
        << expr->getSourceRange();

  return unary;
}

std::shared_ptr<gsl::Expr> ClangASTExprResolver::resolveAssignmentOp(clang::Expr* expr) {
  if(clang::CXXOperatorCallExpr* e = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    GSL_ASSERT(e->getNumArgs() == 2);
    return std::make_shared<gsl::AssignmentExpr>(resolve(e->getArg(0)), resolve(e->getArg(1)),
                                                 clang::getOperatorSpelling(e->getOperator()),
                                                 getSourceLocation(e));
  } else if(clang::BinaryOperator* e = clang::dyn_cast<clang::BinaryOperator>(expr)) {
    return std::make_shared<gsl::AssignmentExpr>(
        resolve(e->getLHS()), resolve(e->getRHS()),
        clang::getOperatorSpelling(clang::BinaryOperator::getOverloadedOperator(e->getOpcode())),
        getSourceLocation(e));
  }
  llvm_unreachable("invalid assignment expr");
}

gsl::BuiltinTypeID ClangASTExprResolver::resolveBuiltinType(clang::Expr* expr) {
  using namespace clang;

  // Resolve C-Style casting
  if(currentCStyleCastExpr_) {
    QualType type = currentCStyleCastExpr_->getType();
    GSL_ASSERT(!type.isNull());

    // Reset (we consumed the casting)
    currentCStyleCastExpr_ = nullptr;

    if(type->isBooleanType()) // bool
      return gsl::BuiltinTypeID::Boolean;
    else if(type->isIntegerType()) // int
      return gsl::BuiltinTypeID::Integer;
    else if(type->isArithmeticType()) // int, float, double...
      return gsl::BuiltinTypeID::Float;
    else
      GSL_LOG(WARNING) << "ignoring C-Style cast to " << type.getAsString();
  }

  // Resolve the builtin type and apply potential casting
  if(isa<IntegerLiteral>(expr)) {
    return gsl::BuiltinTypeID::Integer;
  } else if(isa<FloatingLiteral>(expr)) {
    return gsl::BuiltinTypeID::Float;
  } else if(isa<CXXBoolLiteralExpr>(expr)) {
    return gsl::BuiltinTypeID::Boolean;
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
