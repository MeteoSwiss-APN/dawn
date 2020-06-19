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

#include "gtclang/Frontend/StencilParser.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringSwitch.h"
#include "gtclang/Frontend/ClangASTStmtResolver.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Support/ASTUtils.h"
#include "gtclang/Support/ClangCompat/EvalResult.h"
#include "gtclang/Support/ClangCompat/SourceLocation.h"
#include "gtclang/Support/FileUtil.h"
#include "clang/AST/AST.h"
#include <numeric>

namespace gtclang {

namespace {

//===------------------------------------------------------------------------------------------===//
//     Interval Level Resolver
//===------------------------------------------------------------------------------------------===//

/// @brief Parse non-builtin interval levels
class IntervalLevelParser {
  StencilParser* parser_;
  clang::VarDecl* varDecl_;

  int builtinLevel_;
  int offset_;
  std::string name_;

public:
  IntervalLevelParser(StencilParser* parser)
      : parser_(parser), varDecl_(nullptr), builtinLevel_(0), offset_(0) {}

  void resolveLevel(clang::DeclRefExpr* declRef) {
    clang::VarDecl* varDecl = clang::dyn_cast<clang::VarDecl>(declRef->getDecl());
    varDecl_ = varDecl;
    name_ = varDecl->getNameAsString();

    DAWN_ASSERT_MSG(varDecl, "expected variable declaration for interval bound");
    resolve(varDecl->getInit());
  }

  /// @brief Get the level of the interval
  int getLevel() const { return builtinLevel_ + offset_; }

  /// @brief Get the name of the interval
  std::string getName() const { return name_; }

private:
  void resolve(clang::CXXConstructExpr* expr) {
    if(expr->getNumArgs()) {
      DAWN_ASSERT(expr->getNumArgs() == 1);
      resolve(expr->getArg(0));
    } else {
      clang::SourceLocation locEnd =
          clang_compat::getEndLoc(*varDecl_).getLocWithOffset(name_.size());
      auto builder = parser_->reportDiagnostic(
          locEnd, Diagnostics::DiagKind::err_interval_custom_missing_init);
      builder << name_;
      builder.AddFixItHint(clang::FixItHint::CreateInsertion(locEnd, " = ..."));
    }
  }

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

    // Parse `k_start` in `k_start+1`
    resolve(expr->getArg(0));

    // Parse `1` in `k_start+1`
    if(IntegerLiteral* integer = dyn_cast<clang::IntegerLiteral>(expr->getArg(1))) {
      int offset = static_cast<int>(integer->getValue().signedRoundToDouble());
      offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
      offset_ = offset;
    } else {

      // If it is not an integer literal, it may still be a constant integer expression
      Expr* arg1 = skipAllImplicitNodes(expr->getArg(1));

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      clang_compat::Expr::EvalResultInt res;

      if(var && var->EvaluateAsInt(res, parser_->getContext()->getASTContext())) {
        int offset = static_cast<int>(clang_compat::Expr::getInt(res));
        offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        offset_ = offset;
      } else {
        parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr->getArg(1)),
                                  Diagnostics::DiagKind::err_interval_custom_not_constexpr)
            << expr->getSourceRange()
            << (builtinLevel_ == dawn::sir::Interval::Start ? "k_start" : "k_end");
      }
    }
  }

  void resolve(clang::DeclRefExpr* expr) {
    llvm::StringRef name = expr->getDecl()->getName();
    if(name == "k_start")
      builtinLevel_ = dawn::sir::Interval::Start;
    else if(name == "k_end")
      builtinLevel_ = dawn::sir::Interval::End;
    else {
      parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                Diagnostics::DiagKind::err_interval_custom_not_builtin)
          << expr->getSourceRange() << name;
      parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                                Diagnostics::DiagKind::note_only_builtin_interval_allowed);
    }
  }

  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in IntervalLevelParser");
    }
    llvm_unreachable("invalid expr");
  }
};

//===------------------------------------------------------------------------------------------===//
//     Interval Resolver
//===------------------------------------------------------------------------------------------===//

/// @brief Extract the interval bounds from a C++11 range based for-loop or from a stencil-function
/// Do-Method argument list
class IntervalResolver {
  StencilParser* parser_;

  dawn::Array2i level_;
  dawn::Array2i offset_;
  int curIndex_;

  std::string verticalIndexName_;

public:
  IntervalResolver(StencilParser* parser)
      : parser_(parser), level_{{0, 0}}, offset_{{0, 0}}, curIndex_(0), verticalIndexName_("k") {}

  /// @brief Resolve a range based for-loop
  ///
  /// for(auto k : {k_start, k_end}) {
  ///   ...
  /// }
  void resolve(clang::CXXForRangeStmt* verticalRegionDecl) {
    DAWN_ASSERT(verticalRegionDecl->getRangeStmt()->isSingleDecl());

    // Extract loop bounds
    clang::VarDecl* initializerListDecl =
        clang::dyn_cast<clang::VarDecl>(verticalRegionDecl->getRangeStmt()->getSingleDecl());
    resolve(initializerListDecl->getInit());

    // Extract loop variable
    verticalIndexName_ = verticalRegionDecl->getLoopVariable()->getNameAsString();
  }

  /// @brief Resolve a range based for-loop
  ///
  /// Do(intervalX k_from = k_start, intervalX k_to = k_end) {
  ///   ...
  /// }
  void resolve(clang::ParmVarDecl* k_from, clang::ParmVarDecl* k_to) {
    auto resolveParameter = [&](clang::ParmVarDecl* param, clang::StringRef name) {
      if(param->getName() != name)
        parser_->reportDiagnostic(
            clang_compat::getBeginLoc(*param),
            Diagnostics::DiagKind::err_stencilfun_do_method_invalid_range_keyword)
            << param->getNameAsString();

      if(!param->hasDefaultArg())
        parser_->reportDiagnostic(clang_compat::getBeginLoc(*param),
                                  Diagnostics::DiagKind::err_stencilfun_do_method_missing_interval)
            << param->getNameAsString();

      resolve(param->getDefaultArg());
      curIndex_ += 1;
    };

    resolveParameter(k_from, "k_from");
    resolveParameter(k_to, "k_to");
  }

  /// @brief Get the SIRInterval
  std::pair<std::shared_ptr<dawn::sir::Interval>, dawn::sir::VerticalRegion::LoopOrderKind>
  getInterval() const {

    // Note that intervals have the the invariant lowerBound <= upperBound. We thus encapsulate the
    // loop order here.
    if((level_[0] + offset_[0]) <= (level_[1] + offset_[1]))
      return std::make_pair(
          std::make_shared<dawn::sir::Interval>(level_[0], level_[1], offset_[0], offset_[1]),
          dawn::sir::VerticalRegion::LoopOrderKind::Forward);
    else
      return std::make_pair(
          std::make_shared<dawn::sir::Interval>(level_[1], level_[0], offset_[1], offset_[0]),
          dawn::sir::VerticalRegion::LoopOrderKind::Backward);
  }

  /// @brief Get vertical index name (i.e loop variable in the range-based for loop)
  const std::string& getVerticalIndexName() const { return verticalIndexName_; }

private:
  void resolve(clang::CXXStdInitializerListExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::InitListExpr* expr) {
    for(clang::Expr* e : expr->inits()) {
      resolve(e);
      curIndex_ += 1;
    }
  }

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

    // Parse `k_start` in `k_start+1`
    resolve(expr->getArg(0));

    // Parse `1` in `k_start+1`
    if(IntegerLiteral* integer = dyn_cast<clang::IntegerLiteral>(expr->getArg(1))) {
      int offset = static_cast<int>(integer->getValue().signedRoundToDouble());
      offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
      offset_[curIndex_] = offset;
    } else {

      // If it is not an integer literal, it may still be a constant integer expression
      Expr* arg1 = skipAllImplicitNodes(expr->getArg(1));

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      clang_compat::Expr::EvalResultInt res;

      if(var && var->EvaluateAsInt(res, parser_->getContext()->getASTContext())) {
        int offset = static_cast<int>(clang_compat::Expr::getInt(res));
        offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        offset_[curIndex_] = offset;
      } else {
        parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr->getArg(1)),
                                  Diagnostics::DiagKind::err_interval_not_constexpr)
            << expr->getSourceRange();
      }
    }
  }

  void resolve(clang::CXXConstructExpr* expr) { resolve(expr->getArg(0)); }

  void resolve(clang::DeclRefExpr* expr) {
    std::string typeStr = expr->getType().getAsString();

    // Type has to be intervalX
    if(!clang::StringRef(typeStr).startswith("struct gtclang::dsl::interval")) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_interval_invalid_type)
          << typeStr;
    }

    llvm::StringRef name = expr->getDecl()->getName();
    if(name == "k_start")
      level_[curIndex_] = dawn::sir::Interval::Start;
    else if(name == "k_end")
      level_[curIndex_] = dawn::sir::Interval::End;
    else {
      // Not a builtin interval, parse it!
      auto levelPair = parser_->getCustomIntervalLevel(name);
      if(levelPair.first)
        level_[curIndex_] = levelPair.second;
      else {
        IntervalLevelParser resolver(parser_);
        resolver.resolveLevel(expr);
        level_[curIndex_] = resolver.getLevel();
        parser_->setCustomIntervalLevel(resolver.getName(), resolver.getLevel());
      }
    }
  }

  void resolve(clang::MemberExpr* expr) {
    std::string typeStr = expr->getType().getAsString();
    parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                              Diagnostics::DiagKind::err_interval_invalid_type)
        << typeStr;
  }

  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(CXXStdInitializerListExpr* e = dyn_cast<CXXStdInitializerListExpr>(expr))
      return resolve(e);
    else if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else if(InitListExpr* e = dyn_cast<InitListExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in IntervalResolver");
    }
    llvm_unreachable("invalid expr");
  }
};

//===------------------------------------------------------------------------------------------===//
//     IterationSpace Resolver
//===------------------------------------------------------------------------------------------===//

/// @brief Extract the interval bounds from a C++11 range based for-loop or from a stencil-function
/// Do-Method argument list
class IterationSpaceResolver {
  StencilParser* parser_;

  std::array<int, 6> level_;
  std::array<int, 6> offset_;
  int curIndex_;

  std::string verticalIndexName_;

public:
  IterationSpaceResolver(StencilParser* parser)
      : parser_(parser), level_(), offset_(), curIndex_(0), verticalIndexName_("k") {}

  /// @brief Resolve a range based for-loop
  ///
  /// for(auto k : {k_start, k_end}) {
  ///   ...
  /// }
  void resolve(clang::CXXForRangeStmt* verticalRegionDecl) {
    DAWN_ASSERT(verticalRegionDecl->getRangeStmt()->isSingleDecl());

    // Extract loop bounds
    clang::VarDecl* initializerListDecl =
        clang::dyn_cast<clang::VarDecl>(verticalRegionDecl->getRangeStmt()->getSingleDecl());
    resolve(initializerListDecl->getInit());

    // Extract loop variable
    verticalIndexName_ = verticalRegionDecl->getLoopVariable()->getNameAsString();
  }

  /// @brief Get the SIRInterval
  std::pair<std::shared_ptr<dawn::sir::Interval>, dawn::sir::VerticalRegion::LoopOrderKind>
  getInterval(int dim) const {

    // Note that intervals have the the invariant lowerBound <= upperBound. We thus encapsulate the
    // loop order here.
    if(dim < 3 && dim >= 0) {
      if((level_[2 * dim] + offset_[2 * dim]) <= (level_[2 * dim + 1] + offset_[2 * dim + 1]))
        return std::make_pair(
            std::make_shared<dawn::sir::Interval>(level_[2 * dim], level_[2 * dim + 1],
                                                  offset_[2 * dim], offset_[2 * dim + 1]),
            dawn::sir::VerticalRegion::LoopOrderKind::Forward);
      else
        return std::make_pair(
            std::make_shared<dawn::sir::Interval>(level_[2 * dim + 1], level_[2 * dim],
                                                  offset_[2 * dim + 1], offset_[2 * dim]),
            dawn::sir::VerticalRegion::LoopOrderKind::Backward);
    } else {
      dawn_unreachable("unknown dimension");
    }
  }

  /// @brief Get vertical index name (i.e loop variable in the range-based for loop)
  const std::string& getVerticalIndexName() const { return verticalIndexName_; }

private:
  void resolve(clang::CXXStdInitializerListExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::InitListExpr* expr) {
    for(clang::Expr* e : expr->inits()) {
      resolve(e);
      curIndex_ += 1;
    }
  }

  void resolve(clang::CXXOperatorCallExpr* expr) {
    using namespace clang;

    // Parse `k_start` in `k_start+1`
    resolve(expr->getArg(0));

    // Parse `1` in `k_start+1`
    if(IntegerLiteral* integer = dyn_cast<clang::IntegerLiteral>(expr->getArg(1))) {
      int offset = static_cast<int>(integer->getValue().signedRoundToDouble());
      offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
      offset_[curIndex_] = offset;
    } else {

      // If it is not an integer literal, it may still be a constant integer expression
      Expr* arg1 = skipAllImplicitNodes(expr->getArg(1));

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      clang_compat::Expr::EvalResultInt res;

      if(var && var->EvaluateAsInt(res, parser_->getContext()->getASTContext())) {
        int offset = static_cast<int>(clang_compat::Expr::getInt(res));
        offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        offset_[curIndex_] = offset;
      } else {
        parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr->getArg(1)),
                                  Diagnostics::DiagKind::err_interval_not_constexpr)
            << expr->getSourceRange();
      }
    }
  }

  void resolve(clang::CXXConstructExpr* expr) { resolve(expr->getArg(0)); }

  void resolve(clang::DeclRefExpr* expr) {
    std::string typeStr = expr->getType().getAsString();

    // Type has to be intervalX
    if(!clang::StringRef(typeStr).startswith("struct gtclang::dsl::interval")) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_interval_invalid_type)
          << typeStr;
    }

    llvm::StringRef name = expr->getDecl()->getName();
    if(name == "k_start" || name == "i_start" || name == "j_start")
      level_[curIndex_] = dawn::sir::Interval::Start;
    else if(name == "k_end" || name == "i_end" || name == "j_end")
      level_[curIndex_] = dawn::sir::Interval::End;
    else {
      // Not a builtin interval, parse it!
      auto levelPair = parser_->getCustomIntervalLevel(name);
      if(levelPair.first)
        level_[curIndex_] = levelPair.second;
      else {
        IntervalLevelParser resolver(parser_);
        resolver.resolveLevel(expr);
        level_[curIndex_] = resolver.getLevel();
        parser_->setCustomIntervalLevel(resolver.getName(), resolver.getLevel());
      }
    }
  }

  void resolve(clang::MemberExpr* expr) {
    std::string typeStr = expr->getType().getAsString();
    parser_->reportDiagnostic(clang_compat::getBeginLoc(*expr),
                              Diagnostics::DiagKind::err_interval_invalid_type)
        << typeStr;
  }

  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore implicit nodes
    expr = skipAllImplicitNodes(expr);

    if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(CXXStdInitializerListExpr* e = dyn_cast<CXXStdInitializerListExpr>(expr))
      return resolve(e);
    else if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else if(InitListExpr* e = dyn_cast<InitListExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in IntervalResolver");
    }
    llvm_unreachable("invalid expr");
  }
};

//===------------------------------------------------------------------------------------------===//
//     Boundary Condition Resolver
//===------------------------------------------------------------------------------------------===//

/// @brief Parse a boundary-condition description
class BoundaryConditionResolver {
  StencilParser* parser_;

  // Name of the stencil-function applied to the boundary points
  std::string functor_;

  // Fields to apply the boundary condition
  std::vector<std::string> fields_;

public:
  BoundaryConditionResolver(StencilParser* parser) : parser_(parser) {}

  void resolve(clang::CXXConstructExpr* boundaryCondition) {
    for(clang::Expr* e : boundaryCondition->arguments()) {
      resolve(e);
    }
  }

  /// @brief Get the name of the functor applies to the boundary points
  const std::string& getFunctor() const { return functor_; }

  /// @brief Fields to apply the boundary condition
  const std::vector<std::string>& getFields() const { return fields_; }

private:
  void resolve(clang::CXXTemporaryObjectExpr* expr) {
    DAWN_ASSERT(functor_.empty());
    functor_ = getClassNameFromConstructExpr(expr);

    // Check the stencil function exists
    if(!parser_->hasStencilFunction(functor_)) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_stencilfun_invalid_call);
      return;
    }

    // Check that there are only storage arguments
    dawn::sir::StencilFunction* stencilFun =
        parser_->getStencilFunctionByName(functor_).second.get();

    for(const auto& arg : stencilFun->Args) {
      if(!dawn::isa<dawn::sir::Field>(arg.get()))
        parser_->reportDiagnostic(expr->getLocation(),
                                  Diagnostics::DiagKind::err_boundary_condition_invalid_functor)
            << functor_ << "only storage arguments allowed";
      return;
    }
  }

  void resolve(clang::CXXStdInitializerListExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::MemberExpr* expr) {
    fields_.push_back(expr->getMemberDecl()->getNameAsString());
  }

  void resolve(clang::DeclRefExpr* expr) {
    parser_->reportDiagnostic(expr->getLocation(),
                              Diagnostics::DiagKind::err_boundary_condition_invalid_type)
        << expr->getType().getAsString() << expr->getDecl()->getNameAsString();
    parser_->reportDiagnostic(expr->getLocation(),
                              Diagnostics::DiagKind::note_boundary_condition_only_storage_allowed);
  }

  void resolve(clang::Expr* expr) {
    using namespace clang;
    // ignore all implicit nodes
    expr = skipAllImplicitNodes(expr);

    // Has to be before `CXXConstructExpr`
    if(CXXTemporaryObjectExpr* e = dyn_cast<CXXTemporaryObjectExpr>(expr))
      return resolve(e);

    if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in IntervalResolver");
    }
    llvm_unreachable("invalid expr");
  }
};

} // anonymous namespace

//===------------------------------------------------------------------------------------------===//
//     StencilParser
//===------------------------------------------------------------------------------------------===//

void StencilParser::ParserRecord::addArgDecl(const std::string& name, clang::FieldDecl* decl) {
  CurrentArgDeclMap.emplace(name, ArgDecl(CurrentArgDeclMap.size(), name, decl));
}

StencilParser::StencilParser(GTClangContext* context, GlobalVariableParser& globalVariableParser)
    : context_(context), globalVariableParser_(globalVariableParser) {}

void StencilParser::parseStencil(clang::CXXRecordDecl* recordDecl, const std::string& name) {
  parseStencilImpl(recordDecl, name, StencilKind::SK_Stencil);
}

void StencilParser::parseStencilFunction(clang::CXXRecordDecl* recordDecl,
                                         const std::string& name) {
  parseStencilImpl(recordDecl, name, StencilKind::SK_StencilFunction);
}

const std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::Stencil>>&
StencilParser::getStencilMap() const {
  return stencilMap_;
}

const std::map<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::StencilFunction>>&
StencilParser::getStencilFunctionMap() const {
  return stencilFunctionMap_;
}

clang::DiagnosticBuilder StencilParser::reportDiagnostic(clang::SourceLocation loc,
                                                         Diagnostics::DiagKind kind) {
  return context_->getDiagnostics().report(loc, kind);
}

std::pair<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::StencilFunction>>
StencilParser::getStencilFunctionByName(const std::string& name) {
  for(auto& stencilFunPair : stencilFunctionMap_)
    if(stencilFunPair.second->Name == name)
      return stencilFunPair;
  return decltype(getStencilFunctionByName(name))(nullptr, nullptr);
}

bool StencilParser::hasStencilFunction(const std::string& name) const {
  for(auto& stencilFunPair : stencilFunctionMap_)
    if(stencilFunPair.second->Name == name)
      return true;
  return false;
}

const StencilParser::ParserRecord* StencilParser::getCurrentParserRecord() const {
  return currentParserRecord_.get();
}

std::pair<bool, int> StencilParser::getCustomIntervalLevel(const std::string& name) const {
  auto it = customIntervalLevel_.find(name);
  return it != customIntervalLevel_.end() ? std::make_pair(true, it->second)
                                          : std::make_pair(false, -1);
}

void StencilParser::setCustomIntervalLevel(const std::string& name, int level) {
  customIntervalLevel_.emplace(name, level);
}

bool StencilParser::isGlobalVariable(const std::string& name) const {
  return globalVariableParser_.has(name);
}

void StencilParser::parseStencilImpl(clang::CXXRecordDecl* recordDecl, const std::string& name,
                                     StencilKind kind) {
  using namespace clang;
  using namespace llvm;

  currentParserRecord_ = std::make_unique<ParserRecord>(kind);
  currentParserRecord_->CurrentCXXRecordDecl = recordDecl;

  if(currentParserRecord_->CurrentKind == SK_Stencil) {
    DAWN_ASSERT(!stencilMap_.count(recordDecl));

    // Construct the sir::Stencil and set the name and location
    currentParserRecord_->CurrentStencil =
        stencilMap_.insert({recordDecl, std::make_shared<dawn::sir::Stencil>()})
            .first->second.get();
    currentParserRecord_->CurrentStencil->Name = name;
    currentParserRecord_->CurrentStencil->Loc = getLocation(recordDecl);
    currentParserRecord_->CurrentStencil->StencilDescAst = dawn::sir::makeAST();

    // Parse the arguments
    for(FieldDecl* field : recordDecl->fields())
      parseStorage(field);

  } else {
    DAWN_ASSERT(!stencilFunctionMap_.count(recordDecl));

    // Construct the sir::StencilFunction and set the name and location
    currentParserRecord_->CurrentStencilFunction =
        stencilFunctionMap_.insert({recordDecl, std::make_shared<dawn::sir::StencilFunction>()})
            .first->second.get();
    currentParserRecord_->CurrentStencilFunction->Name = name;
    currentParserRecord_->CurrentStencilFunction->Loc = getLocation(recordDecl);

    // Parse the arguments
    for(FieldDecl* arg : recordDecl->fields())
      parseArgument(arg);
  }

  if(context_->getDiagnosticsEngine().hasErrorOccurred())
    return;

  // Iterate methods
  bool hasDoMethod = false;
  for(CXXMethodDecl* method : recordDecl->methods()) {

    // Skip constructors/destructor and copy/move assignment operators
    if(isa<CXXConstructorDecl>(method) || isa<CXXDestructorDecl>(method) ||
       method->isCopyAssignmentOperator() || method->isMoveAssignmentOperator())
      continue;

    // Parse Do-Method `void Do()` or `double Do()`
    if(method->getNameAsString() == "Do") {
      hasDoMethod = true;

      if(currentParserRecord_->CurrentKind == SK_Stencil)
        parseStencilDoMethod(method);
      else
        parseStencilFunctionDoMethod(method);
    }
    // Parse the generated function that wraps all the boundary conditions
    if(method->getNameAsString() == "__boundary_condition__generated__") {
      auto allBoundaryConditions = parseBoundaryConditions(method);
      for(const auto& boundayCondition : allBoundaryConditions) {
        currentParserRecord_->CurrentStencil->StencilDescAst->getRoot()->push_back(
            boundayCondition);
      }
    }
  }

  // We haven't found a Do method, the stencil is considered ill-formed!
  if(!hasDoMethod) {
    reportDiagnostic(recordDecl->getLocation(), Diagnostics::DiagKind::err_do_method_missing)
        << name;
  }
}

void StencilParser::parseStorage(clang::FieldDecl* field) {
  using namespace dawn::sir;
  DAWN_ASSERT(currentParserRecord_->CurrentKind == SK_Stencil);

  auto name = field->getDeclName().getAsString();

  std::string typeStr;
  if(clang::CXXRecordDecl* decl = field->getType()->getAsCXXRecordDecl())
    typeStr = decl->getNameAsString();
  else
    typeStr = field->getType().getAsString();

  if(typeStr.find("storage") != std::string::npos) {

    DAWN_LOG(INFO) << "Parsing field: " << name;
    auto fieldDimensions = dawn::StringSwitch<std::array<bool, 3>>(typeStr)
                               .Case("storage", {{true, true, true}})
                               .Case("storage_i", {{true, false, false}})
                               .Case("storage_j", {{false, true, false}})
                               .Case("storage_k", {{false, false, true}})
                               .Case("storage_ij", {{true, true, false}})
                               .Case("storage_ik", {{true, false, true}})
                               .Case("storage_jk", {{false, true, true}})
                               .Case("storage_ijk", {{true, true, true}})
                               .Default({{false, false, false}});
    auto SIRField = std::make_shared<Field>(
        name,
        dawn::sir::FieldDimensions(
            dawn::sir::HorizontalFieldDimension(dawn::ast::cartesian,
                                                {fieldDimensions[0], fieldDimensions[1]}),
            fieldDimensions[2]),
        getLocation(field));
    SIRField->IsTemporary = false;
    currentParserRecord_->CurrentStencil->Fields.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, field);

  } else if(typeStr == "var") {

    DAWN_LOG(INFO) << "Parsing temporary field: " << name;
    auto SIRField = std::make_shared<dawn::sir::Field>(
        name,
        dawn::sir::FieldDimensions(
            dawn::sir::HorizontalFieldDimension(dawn::ast::cartesian, {true, true}), true),
        getLocation(field));
    SIRField->IsTemporary = true;
    currentParserRecord_->CurrentStencil->Fields.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, field);

  } else {

    reportDiagnostic(clang_compat::getBeginLoc(*field),
                     Diagnostics::DiagKind::err_stencil_invalid_storage_decl)
        << field->getType().getAsString() << name;
    reportDiagnostic(clang_compat::getBeginLoc(*field),
                     Diagnostics::DiagKind::note_only_storages_allowed);
  }
}

void StencilParser::parseArgument(clang::FieldDecl* arg) {
  DAWN_ASSERT(currentParserRecord_->CurrentKind == SK_StencilFunction);

  auto name = arg->getDeclName().getAsString();

  std::string typeStr;
  if(clang::CXXRecordDecl* decl = arg->getType()->getAsCXXRecordDecl())
    typeStr = decl->getNameAsString();
  else
    typeStr = arg->getType().getAsString();

  if(typeStr == "storage") {

    DAWN_LOG(INFO) << "Parsing field: " << name;
    auto fieldDimensions = dawn::StringSwitch<std::array<bool, 3>>(typeStr)
                               .Case("storage", {{true, true, true}})
                               .Case("storage_i", {{true, false, false}})
                               .Case("storage_j", {{false, true, false}})
                               .Case("storage_k", {{false, false, true}})
                               .Case("storage_ij", {{true, true, false}})
                               .Case("storage_ik", {{true, false, true}})
                               .Case("storage_jk", {{false, true, true}})
                               .Case("storage_ijk", {{true, true, true}})
                               .Default({{false, false, false}});

    auto SIRField = std::make_shared<dawn::sir::Field>(
        name,
        dawn::sir::FieldDimensions(
            dawn::sir::HorizontalFieldDimension(dawn::ast::cartesian,
                                                {fieldDimensions[0], fieldDimensions[1]}),
            fieldDimensions[2]),
        getLocation(arg));
    SIRField->IsTemporary = false;
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, arg);

  } else if(typeStr == "var") {

    DAWN_LOG(INFO) << "Parsing temporary field: " << name;

    reportDiagnostic(clang_compat::getBeginLoc(*arg),
                     Diagnostics::DiagKind::err_stencilfun_invalid_argument_type)
        << typeStr << name;

  } else if(typeStr == "offset") {

    DAWN_LOG(INFO) << "Parsing offset: " << name;
    auto SIROffset = std::make_shared<dawn::sir::Offset>(name, getLocation(arg));
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIROffset);
    currentParserRecord_->addArgDecl(name, arg);

  } else if(typeStr == "direction") {

    DAWN_LOG(INFO) << "Parsing direction: " << name;
    auto SIRDirection = std::make_shared<dawn::sir::Direction>(name, getLocation(arg));
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIRDirection);
    currentParserRecord_->addArgDecl(name, arg);

  } else {
    reportDiagnostic(clang_compat::getBeginLoc(*arg),
                     Diagnostics::DiagKind::err_stencilfun_invalid_argument_type)
        << typeStr << name;
  }
}

void StencilParser::parseStencilFunctionDoMethod(clang::CXXMethodDecl* DoMethod) {
  using namespace clang;
  using namespace llvm;

  DAWN_LOG(INFO)
      << "Parsing stencil-function Do-Method at "
      << getFilename(DoMethod->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  std::shared_ptr<dawn::sir::Interval> intervals = nullptr;
  if(DoMethod->getNumParams() > 0) {
    if(DoMethod->getNumParams() != 2) {
      reportDiagnostic(clang_compat::getBeginLoc(*DoMethod),
                       Diagnostics::DiagKind::err_stencilfun_do_method_invalid_num_arg)
          << DoMethod->getNumParams();
      return;
    }

    IntervalResolver resolver(this);
    resolver.resolve(DoMethod->parameters()[0], DoMethod->parameters()[1]);
    intervals = resolver.getInterval().first;
  }

  if(DoMethod->hasBody()) {

    ClangASTStmtResolver stmtResolver(context_, this);
    auto SIRBlockStmt = dawn::sir::makeBlockStmt(getLocation(DoMethod->getBody()));

    if(CompoundStmt* bodyStmt = dyn_cast<CompoundStmt>(DoMethod->getBody())) {

      // DoMethod is a CompoundStmt, start iterating
      for(Stmt* stmt : bodyStmt->body()) {
        // ignore implicit nodes
        stmt = skipAllImplicitNodes(stmt);

        // Stmt is a range-based for loop which corresponds to a vertical region (not allowed here!)
        if(isa<CXXForRangeStmt>(stmt)) {
          reportDiagnostic(clang_compat::getBeginLoc(*stmt),
                           Diagnostics::DiagKind::err_stencilfun_vertical_region);
          break;
        } else {
          // Resolve statements
          SIRBlockStmt->insert_back(
              stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilBody));
        }
      }
    }

    // Assemble the stencil function
    currentParserRecord_->CurrentStencilFunction->Asts.emplace_back(
        std::make_shared<dawn::sir::AST>(std::move(SIRBlockStmt)));

    if(intervals)
      currentParserRecord_->CurrentStencilFunction->Intervals.emplace_back(std::move(intervals));

  } else {
    // DoMethod is ill-formed (not a compount statement)
    reportDiagnostic(DoMethod->getLocation(), Diagnostics::DiagKind::err_do_method_ill_formed);
  }

  DAWN_LOG(INFO) << "Done parsing stencil-function Do-Method";
}

void StencilParser::parseStencilDoMethod(clang::CXXMethodDecl* DoMethod) {
  using namespace clang;
  using namespace llvm;

  DAWN_LOG(INFO)
      << "Parsing Do-Method at "
      << getFilename(DoMethod->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  if(DoMethod->hasBody()) {
    if(CompoundStmt* bodyStmt = dyn_cast<CompoundStmt>(DoMethod->getBody())) {

      ClangASTStmtResolver stmtResolver(context_, this);

      std::shared_ptr<dawn::sir::AST>& stencilDescAst =
          currentParserRecord_->CurrentStencil->StencilDescAst;

      // DoMethod is a CompoundStmt, start iterating
      for(Stmt* stmt : bodyStmt->body()) {
        // ignore implicit nodes
        stmt = skipAllImplicitNodes(stmt);

        if(CXXForRangeStmt* s = dyn_cast<CXXForRangeStmt>(stmt)) {
          // stmt is a range-based for loop which corresponds to a VerticalRegion
          if(s->getLoopVariable()->getName() == "__k_indexrange__") {
            stencilDescAst->getRoot()->push_back(parseIterationSpace(s));
          } else {
            stencilDescAst->getRoot()->push_back(parseVerticalRegion(s));
          }

        } else if(CXXConstructExpr* s = dyn_cast<CXXConstructExpr>(stmt)) {

          if(getClassNameFromConstructExpr(s) == "boundary_condition")
            // stmt is a declaration of a boundary condition
            stencilDescAst->getRoot()->push_back(parseBoundaryCondition(s));
          else if(std::shared_ptr<dawn::sir::StencilCallDeclStmt> stencilCall = parseStencilCall(s))
            // stmt is a call to another stencil
            stencilDescAst->getRoot()->push_back(stencilCall);

        } else if(isa<DeclStmt>(stmt)) {
          // smt is a local variable declaration (e.g `double a = ...`)
          stencilDescAst->getRoot()->insert_back(
              stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilDesc));

        } else if(isa<BinaryOperator>(stmt)) {
          // stmt is a local variable assignment (e.g `a = ...`)
          stencilDescAst->getRoot()->insert_back(
              stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilDesc));

        } else if(isa<IfStmt>(stmt)) {
          // smt is an if-statement (e.g `if(...) { ... }`)
          stencilDescAst->getRoot()->insert_back(
              stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilDesc));

        } else {
          // Not a valid statement inside a Do-Method
          reportDiagnostic(clang_compat::getBeginLoc(*stmt),
                           Diagnostics::DiagKind::err_do_method_illegal_stmt);
        }
      }

    } else {
      // DoMethod is ill-formed (not a compount statement)
      reportDiagnostic(DoMethod->getLocation(), Diagnostics::DiagKind::err_do_method_ill_formed);
    }
  }

  DAWN_LOG(INFO) << "Done parsing Do-Method";
}

std::shared_ptr<dawn::sir::StencilCallDeclStmt>
StencilParser::parseStencilCall(clang::CXXConstructExpr* stencilCall) {

  std::string callee = getClassNameFromConstructExpr(stencilCall);

  DAWN_LOG(INFO)
      << "Parsing stencil-call at "
      << getFilename(stencilCall->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  // Check if stencil exists
  auto stencilIt = std::find_if(
      stencilMap_.begin(), stencilMap_.end(),
      [&](const std::pair<clang::CXXRecordDecl*, std::shared_ptr<dawn::sir::Stencil>>& spair) {
        return spair.second->Name == callee;
      });

  if(stencilIt == stencilMap_.end()) {
    reportDiagnostic(stencilCall->getLocation(),
                     Diagnostics::DiagKind::err_stencilcall_invalid_call)
        << stencilCall->getSourceRange() << callee;
    return nullptr;
  }

  // Check we don't call ourselves (would result in infinite recursion)
  if(callee == currentParserRecord_->CurrentStencil->Name) {
    reportDiagnostic(stencilCall->getLocation(),
                     Diagnostics::DiagKind::err_stencilcall_invalid_call)
        << stencilCall->getSourceRange() << callee;

    std::string msg = "self recursive calls result in infinite recursion";
    reportDiagnostic(stencilIt->first->getLocation(),
                     Diagnostics::DiagKind::note_stencilcall_not_viable)
        << msg;
    return nullptr;
  }

  std::vector<std::string> fieldNames;

  // Check if arguments are of type `gtclang::dsl::storage` and the number of arguments match
  for(auto* arg : stencilCall->arguments()) {
    if(clang::MemberExpr* member = clang::dyn_cast<clang::MemberExpr>(arg)) {
      std::string type = member->getMemberDecl()->getType().getAsString();
      std::string declType = member->getType()->getAsCXXRecordDecl()->getName();

      if((declType.find("storage") == std::string::npos) && declType != "var") {
        reportDiagnostic(clang_compat::getBeginLoc(*member),
                         Diagnostics::DiagKind::err_stencilcall_invalid_argument_type)
            << member->getSourceRange() << type << callee;
        reportDiagnostic(clang_compat::getBeginLoc(*member),
                         Diagnostics::DiagKind::note_stencilcall_only_storage_allowed);
        return nullptr;
      }

      std::string name = member->getMemberDecl()->getNameAsString();
      fieldNames.push_back(name);
    } else {
      DAWN_ASSERT_MSG(0, "expected `clang::MemberExpr` in argument-list of stencil call");
    }
  }

  // Check if number of arguments match. Note that temporaries don't need to be provided, we
  // implicitly generate a new temporary for each temporary storage in the stencil
  std::size_t parsedArgs = fieldNames.size();

  const auto& requiredFields = stencilIt->second->Fields;
  std::size_t requiredArgs =
      std::accumulate(requiredFields.begin(), requiredFields.end(), 0,
                      [](std::size_t acc, const std::shared_ptr<dawn::sir::Field>& field) {
                        return acc + !field->IsTemporary;
                      });

  if(parsedArgs != requiredArgs) {
    reportDiagnostic(stencilCall->getLocation(),
                     Diagnostics::DiagKind::err_stencilcall_invalid_call)
        << callee;

    std::stringstream ss;
    ss << "requires " << requiredArgs << " argument" << (requiredArgs == 1 ? "" : "s") << ", but "
       << parsedArgs << " " << (parsedArgs == 1 ? "was" : "were") << " provided";

    reportDiagnostic(stencilIt->first->getLocation(),
                     Diagnostics::DiagKind::note_stencilcall_not_viable)
        << ss.str();
    return nullptr;
  }

  auto astStencilCall = std::make_shared<dawn::ast::StencilCall>(callee, getLocation(stencilCall));
  astStencilCall->Args = std::move(fieldNames);

  DAWN_LOG(INFO) << "Done parsing stencil call";
  return dawn::sir::makeStencilCallDeclStmt(astStencilCall, astStencilCall->Loc);
}

std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>
StencilParser::parseVerticalRegion(clang::CXXForRangeStmt* verticalRegion) {
  using namespace clang;
  using namespace llvm;

  DAWN_LOG(INFO) << "Parsing vertical region at " << getLocation(verticalRegion);

  // The idea is to translate each vertical region, given as a C++11 range-based for loop (i.e
  // `for(auto k : {X,X})`) into a VerticalRegionDeclStmt

  // Extract the Interval from the loop bounds
  IntervalResolver intervalResolver(this);
  intervalResolver.resolve(verticalRegion);

  // Extract the Do-Method body (AST) from the loop body
  auto SIRBlockStmt = dawn::sir::makeBlockStmt(getLocation(verticalRegion));

  // There is a difference between
  //
  // for(...)  and   for(...) {
  //   XXX              XXX
  //                 }
  //
  // In the former case we direclty get the one Stmt node, while in the latter we get a CompountStmt
  if(verticalRegion->getBody()) {

    ClangASTStmtResolver stmtResolver(context_, this);

    if(CompoundStmt* compoundStmt = dyn_cast<CompoundStmt>(verticalRegion->getBody())) {
      // Case for(...) { XXX }
      for(Stmt* stmt : compoundStmt->body()) {
        // ignore implicit nodes
        stmt = skipAllImplicitNodes(stmt);

        SIRBlockStmt->insert_back(
            stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilBody));
      }

    } else {
      // Case for(...) XXX
      SIRBlockStmt->insert_back(stmtResolver.resolveStmt(verticalRegion->getBody(),
                                                         ClangASTStmtResolver::AK_StencilBody));
    }
  }

  // Assemble the vertical region and register it within the current Stencil
  auto SIRAST = std::make_shared<dawn::sir::AST>(std::move(SIRBlockStmt));

  auto intervalPair = intervalResolver.getInterval();

  auto SIRVerticalRegion = std::make_shared<dawn::sir::VerticalRegion>(
      SIRAST, intervalPair.first, intervalPair.second, getLocation(verticalRegion));

  DAWN_LOG(INFO) << "Done parsing vertical region";
  return dawn::sir::makeVerticalRegionDeclStmt(SIRVerticalRegion, SIRVerticalRegion->Loc);
}

std::shared_ptr<dawn::sir::VerticalRegionDeclStmt>
StencilParser::parseIterationSpace(clang::CXXForRangeStmt* iterationSpaceDecl) {
  using namespace clang;
  using namespace llvm;

  DAWN_LOG(INFO) << "Parsing iteraion space at " << getLocation(iterationSpaceDecl);

  // The idea is to translate each iteration space, given as a C++11 range-based for loop (i.e
  // `for(auto __k_indexrange__ : {X,X,Y,Y,Z,Z})`) into a VerticalRegionDeclStmt with the set bounds

  // Extract the Interval from the loop bounds
  IterationSpaceResolver intervalResolver(this);
  intervalResolver.resolve(iterationSpaceDecl);

  // Extract the Do-Method body (AST) from the loop body
  auto SIRBlockStmt = dawn::sir::makeBlockStmt(getLocation(iterationSpaceDecl));

  // There is a difference between
  //
  // for(...)  and   for(...) {
  //   XXX              XXX
  //                 }
  //
  // In the former case we direclty get the one Stmt node, while in the latter we get a CompountStmt
  if(iterationSpaceDecl->getBody()) {

    ClangASTStmtResolver stmtResolver(context_, this);

    if(CompoundStmt* compoundStmt = dyn_cast<CompoundStmt>(iterationSpaceDecl->getBody())) {
      // Case for(...) { XXX }
      for(Stmt* stmt : compoundStmt->body()) {
        // ignore implicit nodes
        stmt = skipAllImplicitNodes(stmt);

        SIRBlockStmt->insert_back(
            stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilBody));
      }

    } else {
      // Case for(...) XXX
      SIRBlockStmt->insert_back(stmtResolver.resolveStmt(iterationSpaceDecl->getBody(),
                                                         ClangASTStmtResolver::AK_StencilBody));
    }
  }

  // Assemble the vertical region and register it within the current Stencil
  auto SIRAST = std::make_shared<dawn::sir::AST>(std::move(SIRBlockStmt));

  auto iInterval = intervalResolver.getInterval(0).first;
  auto jInterval = intervalResolver.getInterval(1).first;
  auto kIntervalPair = intervalResolver.getInterval(2);

  auto SIRVerticalRegion = std::make_shared<dawn::sir::VerticalRegion>(
      SIRAST, kIntervalPair.first, kIntervalPair.second, getLocation(iterationSpaceDecl));
  if(iInterval->LowerLevel != dawn::sir::Interval::Start || iInterval->LowerOffset != 0 ||
     iInterval->UpperLevel != dawn::sir::Interval::End || iInterval->UpperOffset != 0) {
    SIRVerticalRegion->IterationSpace[0] = *iInterval;
  }
  if(jInterval->LowerLevel != dawn::sir::Interval::Start || jInterval->LowerOffset != 0 ||
     jInterval->UpperLevel != dawn::sir::Interval::End || jInterval->UpperOffset != 0) {
    SIRVerticalRegion->IterationSpace[1] = *jInterval;
  }

  DAWN_LOG(INFO) << "Done parsing iteration space";
  return dawn::sir::makeVerticalRegionDeclStmt(SIRVerticalRegion, SIRVerticalRegion->Loc);
}

std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt>
StencilParser::parseBoundaryCondition(clang::CXXConstructExpr* boundaryCondition) {
  using namespace clang;
  using namespace llvm;

  DAWN_ASSERT(getClassNameFromConstructExpr(boundaryCondition) == "boundary_condition");

  DAWN_LOG(INFO) << "Parsing boundary-condition at "
                 << getFilename(boundaryCondition->getLocation().printToString(
                                    context_->getSourceManager()))
                        .str()
                 << " ...";

  BoundaryConditionResolver resolver(this);
  resolver.resolve(boundaryCondition);

  auto ASTBoundaryCondition = dawn::sir::makeBoundaryConditionDeclStmt(
      resolver.getFunctor(), getLocation(boundaryCondition));

  for(const auto& name : resolver.getFields()) {
    dawn::sir::Stencil* curStencil = currentParserRecord_->CurrentStencil;

    auto it = std::find_if(
        curStencil->Fields.begin(), curStencil->Fields.end(),
        [&name](const std::shared_ptr<dawn::sir::Field>& field) { return field->Name == name; });

    if(it == curStencil->Fields.end()) {
      auto builder = reportDiagnostic(boundaryCondition->getLocation(),
                                      Diagnostics::DiagKind::err_boundary_condition_invalid_arg);
      builder << name;
    } else {
      ASTBoundaryCondition->getFields().push_back((*it)->Name);
    }
  }

  DAWN_LOG(INFO) << "Done parsing boundary-condition";
  return ASTBoundaryCondition;
}

std::vector<std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt>>
StencilParser::parseBoundaryConditions(clang::CXXMethodDecl* allBoundaryConditions) {
  using namespace clang;
  using namespace llvm;
  DAWN_LOG(INFO) << "Parsing all the boundary conditions at " << getLocation(allBoundaryConditions);

  std::vector<std::shared_ptr<dawn::sir::BoundaryConditionDeclStmt>> parsedBoundayConditions;
  CompoundStmt* bodyStmt = dyn_cast<CompoundStmt>(allBoundaryConditions->getBody());

  // loop over all the bounary condition stmts
  // they all share the same signature:
  //    boundary_condition(functor(), Field [, arguments...]);
  for(Stmt* stmt : bodyStmt->body()) {
    // ignore implicit nodes
    stmt = skipAllImplicitNodes(stmt);
    if(CXXTemporaryObjectExpr* temporary = dyn_cast<CXXTemporaryObjectExpr>(stmt)) {
      if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(temporary)) {
        // Resolve the statement
        BoundaryConditionResolver res(this);
        res.resolve(e);

        // create that DeclStmt with the functor and add the Fields / Arugments, where the field to
        // apply to is at Field[0] and all the arguments follow
        auto bc = dawn::sir::makeBoundaryConditionDeclStmt(res.getFunctor());
        for(const auto& fieldName : res.getFields()) {
          bc->getFields().emplace_back(fieldName);
        }
        parsedBoundayConditions.push_back(bc);
      } else {
        e->dumpColor();
        DAWN_ASSERT_MSG(0, "unresolved expression in Bounday-Condition Parser");
      }

    } else {
      stmt->dumpColor();
      DAWN_ASSERT_MSG(0, "unresolved expression in Bounday-Condition Parser");
    }
  }
  return parsedBoundayConditions;
}

dawn::SourceLocation StencilParser::getLocation(clang::Decl* decl) const {
  clang::PresumedLoc ploc = context_->getSourceManager().getPresumedLoc(decl->getLocation());
  return dawn::SourceLocation(ploc.getLine(), ploc.getColumn());
}

dawn::SourceLocation StencilParser::getLocation(clang::Stmt* stmt) const {
  clang::PresumedLoc ploc =
      context_->getSourceManager().getPresumedLoc(clang_compat::getBeginLoc(*stmt));
  return dawn::SourceLocation(ploc.getLine(), ploc.getColumn());
}

} // namespace gtclang
