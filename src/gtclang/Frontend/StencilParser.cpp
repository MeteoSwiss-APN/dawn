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

#include "gsl/SIR/SIR.h"
#include "gsl/Support/Array.h"
#include "gsl/Support/Assert.h"
#include "gsl/Support/Casting.h"
#include "gsl/Support/Logging.h"
#include "gtclang/Frontend/ClangASTStmtResolver.h"
#include "gtclang/Frontend/GTClangContext.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Frontend/StencilParser.h"
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

    GSL_ASSERT_MSG(varDecl, "expected variable declaration for interval bound");
    resolve(varDecl->getInit());
  }

  /// @brief Get the level of the interval
  int getLevel() const { return builtinLevel_ + offset_; }

  /// @brief Get the name of the interval
  std::string getName() const { return name_; }

private:
  void resolve(clang::CXXConstructExpr* expr) {
    if(expr->getNumArgs()) {
      GSL_ASSERT(expr->getNumArgs() == 1);
      resolve(expr->getArg(0));
    } else {
      clang::SourceLocation locEnd = varDecl_->getLocEnd().getLocWithOffset(name_.size());
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
      Expr* arg1 = expr->getArg(1);
      if(ImplicitCastExpr* castExpr = dyn_cast<ImplicitCastExpr>(arg1))
        arg1 = castExpr->getSubExpr();

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      llvm::APSInt res;

      if(var && var->EvaluateAsInt(res, parser_->getContext()->getASTContext())) {
        int offset = static_cast<int>(res.getExtValue());
        offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        offset_ = offset;
      } else {
        parser_->reportDiagnostic(expr->getArg(1)->getLocStart(),
                                  Diagnostics::DiagKind::err_interval_custom_not_constexpr)
            << expr->getSourceRange()
            << (builtinLevel_ == gsl::sir::Interval::Start ? "k_start" : "k_end");
      }
    }
  }

  void resolve(clang::DeclRefExpr* expr) {
    llvm::StringRef name = expr->getDecl()->getName();
    if(name == "k_start")
      builtinLevel_ = gsl::sir::Interval::Start;
    else if(name == "k_end")
      builtinLevel_ = gsl::sir::Interval::End;
    else {
      parser_->reportDiagnostic(expr->getLocStart(),
                                Diagnostics::DiagKind::err_interval_custom_not_builtin)
          << expr->getSourceRange() << name;
      parser_->reportDiagnostic(expr->getLocStart(),
                                Diagnostics::DiagKind::note_only_builtin_interval_allowed);
    }
  }

  void resolve(clang::MaterializeTemporaryExpr* expr) { resolve(expr->GetTemporaryExpr()); }

  void resolve(clang::ImplicitCastExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::Expr* expr) {
    using namespace clang;

    if(CXXOperatorCallExpr* e = dyn_cast<CXXOperatorCallExpr>(expr))
      return resolve(e);
    else if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else if(ImplicitCastExpr* e = dyn_cast<ImplicitCastExpr>(expr))
      return resolve(e);
    else if(MaterializeTemporaryExpr* e = dyn_cast<MaterializeTemporaryExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      GSL_ASSERT_MSG(0, "unresolved expression in IntervalLevelParser");
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

  gsl::Array2i level_;
  gsl::Array2i offset_;
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
    GSL_ASSERT(verticalRegionDecl->getRangeStmt()->isSingleDecl());

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
            param->getLocStart(),
            Diagnostics::DiagKind::err_stencilfun_do_method_invalid_range_keyword)
            << param->getNameAsString();

      if(!param->hasDefaultArg())
        parser_->reportDiagnostic(param->getLocStart(),
                                  Diagnostics::DiagKind::err_stencilfun_do_method_missing_interval)
            << param->getNameAsString();

      resolve(param->getDefaultArg());
      curIndex_ += 1;
    };

    resolveParameter(k_from, "k_from");
    resolveParameter(k_to, "k_to");
  }

  /// @brief Get the SIRInterval
  std::pair<std::shared_ptr<gsl::sir::Interval>, gsl::sir::VerticalRegion::LoopOrderKind>
  getInterval() const {

    // Note that intervals have the the invariant lowerBound <= upperBound. We thus encapsulate the
    // loop order here.
    if((level_[0] + offset_[0]) <= (level_[1] + offset_[1]))
      return std::make_pair(
          std::make_shared<gsl::sir::Interval>(level_[0], level_[1], offset_[0], offset_[1]),
          gsl::sir::VerticalRegion::LK_Forward);
    else
      return std::make_pair(
          std::make_shared<gsl::sir::Interval>(level_[1], level_[0], offset_[1], offset_[0]),
          gsl::sir::VerticalRegion::LK_Backward);
  }

  /// @brief Get vertical index name (i.e loop variable in the range-based for loop)
  const std::string& getVerticalIndexName() const { return verticalIndexName_; }

private:
  void resolve(clang::MaterializeTemporaryExpr* expr) { resolve(expr->GetTemporaryExpr()); }

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
      Expr* arg1 = expr->getArg(1);
      if(ImplicitCastExpr* castExpr = dyn_cast<ImplicitCastExpr>(arg1))
        arg1 = castExpr->getSubExpr();

      DeclRefExpr* var = dyn_cast<DeclRefExpr>(arg1);
      llvm::APSInt res;

      if(var && var->EvaluateAsInt(res, parser_->getContext()->getASTContext())) {
        int offset = static_cast<int>(res.getExtValue());
        offset *= expr->getOperator() == clang::OO_Minus ? -1 : 1;
        offset_[curIndex_] = offset;
      } else {
        parser_->reportDiagnostic(expr->getArg(1)->getLocStart(),
                                  Diagnostics::DiagKind::err_interval_not_constexpr)
            << expr->getSourceRange();
      }
    }
  }

  void resolve(clang::CXXConstructExpr* expr) { resolve(expr->getArg(0)); }

  void resolve(clang::ImplicitCastExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::DeclRefExpr* expr) {
    std::string typeStr = expr->getType().getAsString();

    // Type has to be intervalX
    if(!clang::StringRef(typeStr).startswith("struct gridtools::clang::interval")) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_interval_invalid_type)
          << typeStr;
    }

    llvm::StringRef name = expr->getDecl()->getName();
    if(name == "k_start")
      level_[curIndex_] = gsl::sir::Interval::Start;
    else if(name == "k_end")
      level_[curIndex_] = gsl::sir::Interval::End;
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
    parser_->reportDiagnostic(expr->getLocStart(), Diagnostics::DiagKind::err_interval_invalid_type)
        << typeStr;
  }

  void resolve(clang::Expr* expr) {
    using namespace clang;

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
    else if(ImplicitCastExpr* e = dyn_cast<ImplicitCastExpr>(expr))
      return resolve(e);
    else if(MaterializeTemporaryExpr* e = dyn_cast<MaterializeTemporaryExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      GSL_ASSERT_MSG(0, "unresolved expression in IntervalResolver");
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
    for(auto e : boundaryCondition->arguments())
      resolve(e);
  }

  /// @brief Get the name of the functor applies to the boundary points
  const std::string& getFunctor() const { return functor_; }

  /// @brief Fields to apply the boundary condition
  const std::vector<std::string>& getFields() const { return fields_; }

private:
  void resolve(clang::MaterializeTemporaryExpr* expr) { resolve(expr->GetTemporaryExpr()); }

  void resolve(clang::CXXTemporaryObjectExpr* expr) {
    GSL_ASSERT(functor_.empty());
    functor_ = expr->getConstructor()->getNameAsString();
  
    // Check the stencil function exists
    if(!parser_->hasStencilFunction(functor_)) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_stencilfun_invalid_call);
      return;
    }
  
    // Check that there are only storage arguments
    gsl::sir::StencilFunction* stencilFun = parser_->getStencilFunctionByName(functor_).second.get();
  
    if(stencilFun->Args.size() > 1) {
      parser_->reportDiagnostic(expr->getLocation(),
                                Diagnostics::DiagKind::err_boundary_condition_invalid_functor)
          << functor_ << "expected single argument";
      return;
    }
  
    for(const auto& arg : stencilFun->Args) {
      if(!gsl::isa<gsl::sir::Field>(arg.get()))
        parser_->reportDiagnostic(expr->getLocation(),
                                  Diagnostics::DiagKind::err_boundary_condition_invalid_functor)
            << functor_ << "only storage arguments allowed";
      return;
    }
  }

  void resolve(clang::CXXStdInitializerListExpr* expr) { resolve(expr->getSubExpr()); }

  void resolve(clang::ImplicitCastExpr* expr) { resolve(expr->getSubExpr()); }

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

    // Has to be before `CXXConstructExpr`
    if(CXXTemporaryObjectExpr* e = dyn_cast<CXXTemporaryObjectExpr>(expr))
      return resolve(e);

    if(CXXConstructExpr* e = dyn_cast<CXXConstructExpr>(expr))
      return resolve(e);
    else if(DeclRefExpr* e = dyn_cast<DeclRefExpr>(expr))
      return resolve(e);
    else if(ImplicitCastExpr* e = dyn_cast<ImplicitCastExpr>(expr))
      return resolve(e);
    else if(MaterializeTemporaryExpr* e = dyn_cast<MaterializeTemporaryExpr>(expr))
      return resolve(e);
    else if(MemberExpr* e = dyn_cast<MemberExpr>(expr))
      return resolve(e);
    else {
      expr->dumpColor();
      GSL_ASSERT_MSG(0, "unresolved expression in IntervalResolver");
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

const std::map<clang::CXXRecordDecl*, std::shared_ptr<gsl::sir::Stencil>>&
StencilParser::getStencilMap() const {
  return stencilMap_;
}

const std::map<clang::CXXRecordDecl*, std::shared_ptr<gsl::sir::StencilFunction>>&
StencilParser::getStencilFunctionMap() const {
  return stencilFunctionMap_;
}

clang::DiagnosticBuilder StencilParser::reportDiagnostic(clang::SourceLocation loc,
                                                         Diagnostics::DiagKind kind) {
  return context_->getDiagnostics().report(loc, kind);
}

std::pair<clang::CXXRecordDecl*, std::shared_ptr<gsl::sir::StencilFunction>>
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

  currentParserRecord_ = llvm::make_unique<ParserRecord>(kind);
  currentParserRecord_->CurrentCXXRecordDecl = recordDecl;

  if(currentParserRecord_->CurrentKind == SK_Stencil) {
    GSL_ASSERT(!stencilMap_.count(recordDecl));

    // Construct the sir::Stencil and set the name and location
    currentParserRecord_->CurrentStencil =
        stencilMap_.insert({recordDecl, std::make_shared<gsl::sir::Stencil>()}).first->second.get();
    currentParserRecord_->CurrentStencil->Name = name;
    currentParserRecord_->CurrentStencil->Loc = getLocation(recordDecl);
    currentParserRecord_->CurrentStencil->StencilDescAst = std::make_shared<gsl::AST>();

    // Parse the arguments
    for(FieldDecl* field : recordDecl->fields())
      parseStorage(field);

  } else {
    GSL_ASSERT(!stencilFunctionMap_.count(recordDecl));

    // Construct the sir::StencilFunction and set the name and location
    currentParserRecord_->CurrentStencilFunction =
        stencilFunctionMap_.insert({recordDecl, std::make_shared<gsl::sir::StencilFunction>()})
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
  }

  // We haven't found a Do method, the stencil is considered ill-formed!
  if(!hasDoMethod) {
    reportDiagnostic(recordDecl->getLocation(), Diagnostics::DiagKind::err_do_method_missing)
        << name;
  }
}

void StencilParser::parseStorage(clang::FieldDecl* field) {
  GSL_ASSERT(currentParserRecord_->CurrentKind == SK_Stencil);

  auto name = field->getDeclName().getAsString();

  std::string typeStr;
  if(clang::CXXRecordDecl* decl = field->getType()->getAsCXXRecordDecl())
    typeStr = decl->getNameAsString();
  else
    typeStr = field->getType().getAsString();

  if(typeStr == "storage") {

    GSL_LOG(INFO) << "Parsing field: " << name;
    auto SIRField = std::make_shared<gsl::sir::Field>(name, getLocation(field));
    SIRField->IsTemporary = false;
    currentParserRecord_->CurrentStencil->Fields.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, field);

  } else if(typeStr == "temporary_storage") {

    GSL_LOG(INFO) << "Parsing temporary field: " << name;
    auto SIRField = std::make_shared<gsl::sir::Field>(name, getLocation(field));
    SIRField->IsTemporary = true;
    currentParserRecord_->CurrentStencil->Fields.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, field);

  } else {

    reportDiagnostic(field->getLocStart(), Diagnostics::DiagKind::err_stencil_invalid_storage_decl)
        << field->getType().getAsString() << name;
    reportDiagnostic(field->getLocStart(), Diagnostics::DiagKind::note_only_storages_allowed);
  }
}

void StencilParser::parseArgument(clang::FieldDecl* arg) {
  GSL_ASSERT(currentParserRecord_->CurrentKind == SK_StencilFunction);

  auto name = arg->getDeclName().getAsString();

  std::string typeStr;
  if(clang::CXXRecordDecl* decl = arg->getType()->getAsCXXRecordDecl())
    typeStr = decl->getNameAsString();
  else
    typeStr = arg->getType().getAsString();

  if(typeStr == "storage") {

    GSL_LOG(INFO) << "Parsing field: " << name;
    auto SIRField = std::make_shared<gsl::sir::Field>(name, getLocation(arg));
    SIRField->IsTemporary = false;
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, arg);

  } else if(typeStr == "temporary_storage") {

    GSL_LOG(INFO) << "Parsing temporary field: " << name;
    auto SIRField = std::make_shared<gsl::sir::Field>(name, getLocation(arg));
    SIRField->IsTemporary = true;
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIRField);
    currentParserRecord_->addArgDecl(name, arg);

  } else if(typeStr == "offset") {

    GSL_LOG(INFO) << "Parsing offset: " << name;
    auto SIROffset = std::make_shared<gsl::sir::Offset>(name, getLocation(arg));
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIROffset);
    currentParserRecord_->addArgDecl(name, arg);

  } else if(typeStr == "direction") {

    GSL_LOG(INFO) << "Parsing direction: " << name;
    auto SIRDirection = std::make_shared<gsl::sir::Direction>(name, getLocation(arg));
    currentParserRecord_->CurrentStencilFunction->Args.emplace_back(SIRDirection);
    currentParserRecord_->addArgDecl(name, arg);

  } else {
    reportDiagnostic(arg->getLocStart(),
                     Diagnostics::DiagKind::err_stencilfun_invalid_argument_type)
        << typeStr << name;
  }
}

void StencilParser::parseStencilFunctionDoMethod(clang::CXXMethodDecl* DoMethod) {
  using namespace clang;
  using namespace llvm;

  GSL_LOG(INFO)
      << "Parsing stencil-function Do-Method at "
      << getFilename(DoMethod->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  std::shared_ptr<gsl::sir::Interval> intervals = nullptr;
  if(DoMethod->getNumParams() > 0) {
    if(DoMethod->getNumParams() != 2) {
      reportDiagnostic(DoMethod->getLocStart(),
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
    auto SIRBlockStmt = std::make_shared<gsl::BlockStmt>(getLocation(DoMethod->getBody()));

    if(CompoundStmt* bodyStmt = dyn_cast<CompoundStmt>(DoMethod->getBody())) {

      // DoMethod is a CompoundStmt, start iterating
      for(Stmt* stmt : bodyStmt->body()) {
        // Stmt is a range-based for loop which corresponds to a vertical region (not allowed here!)
        if(isa<CXXForRangeStmt>(stmt)) {
          reportDiagnostic(stmt->getLocStart(),
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
        std::make_shared<gsl::AST>(std::move(SIRBlockStmt)));

    if(intervals)
      currentParserRecord_->CurrentStencilFunction->Intervals.emplace_back(std::move(intervals));

  } else {
    // DoMethod is ill-formed (not a compount statement)
    reportDiagnostic(DoMethod->getLocation(), Diagnostics::DiagKind::err_do_method_ill_formed);
  }

  GSL_LOG(INFO) << "Done parsing stencil-function Do-Method";
}

void StencilParser::parseStencilDoMethod(clang::CXXMethodDecl* DoMethod) {
  using namespace clang;
  using namespace llvm;

  GSL_LOG(INFO)
      << "Parsing Do-Method at "
      << getFilename(DoMethod->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  if(DoMethod->hasBody()) {
    if(CompoundStmt* bodyStmt = dyn_cast<CompoundStmt>(DoMethod->getBody())) {

      ClangASTStmtResolver stmtResolver(context_, this);

      std::shared_ptr<gsl::AST>& stencilDescAst =
          currentParserRecord_->CurrentStencil->StencilDescAst;

      // DoMethod is a CompoundStmt, start iterating
      for(Stmt* stmt : bodyStmt->body()) {

        if(CXXForRangeStmt* s = dyn_cast<CXXForRangeStmt>(stmt)) {
          // stmt is a range-based for loop which corresponds to a VerticalRegion
          stencilDescAst->getRoot()->push_back(parseVerticalRegion(s));

        } else if(CXXConstructExpr* s = dyn_cast<CXXConstructExpr>(stmt)) {

          if(s->getConstructor()->getNameAsString() == "boundary_condition")
            // stmt is a declaration of a boundary condition
            stencilDescAst->getRoot()->push_back(parseBoundaryCondition(s));
          else
            // stmt is a call to another stencil
            stencilDescAst->getRoot()->push_back(parseStencilCall(s));

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
          reportDiagnostic(stmt->getLocStart(), Diagnostics::DiagKind::err_do_method_illegal_stmt);
        }
      }

    } else {
      // DoMethod is ill-formed (not a compount statement)
      reportDiagnostic(DoMethod->getLocation(), Diagnostics::DiagKind::err_do_method_ill_formed);
    }
  }

  GSL_LOG(INFO) << "Done parsing Do-Method";
}

std::shared_ptr<gsl::StencilCallDeclStmt>
StencilParser::parseStencilCall(clang::CXXConstructExpr* stencilCall) {
  std::string callee = stencilCall->getConstructor()->getNameAsString();

  GSL_LOG(INFO)
      << "Parsing stencil-call at "
      << getFilename(stencilCall->getLocation().printToString(context_->getSourceManager())).str()
      << " ...";

  // Check if stencil exists
  auto stencilIt = std::find_if(
      stencilMap_.begin(), stencilMap_.end(),
      [&](const std::pair<clang::CXXRecordDecl*, std::shared_ptr<gsl::sir::Stencil>>& spair) {
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

  std::vector<std::shared_ptr<gsl::sir::Field>> fields;

  // Check if arguments are of type `gridtools::clang::storage` and the number of arguments match
  for(auto* arg : stencilCall->arguments()) {
    if(clang::MemberExpr* member = clang::dyn_cast<clang::MemberExpr>(arg)) {
      std::string type = member->getMemberDecl()->getType().getAsString();
      std::string declType = member->getType()->getAsCXXRecordDecl()->getName();

      if(declType != "storage" && declType != "temporary_storage") {
        reportDiagnostic(member->getLocStart(),
                         Diagnostics::DiagKind::err_stencilcall_invalid_argument_type)
            << member->getSourceRange() << type << callee;
        reportDiagnostic(member->getLocStart(),
                         Diagnostics::DiagKind::note_stencilcall_only_storage_allowed);
        return nullptr;
      }

      std::string name = member->getMemberDecl()->getNameAsString();
      fields.push_back(std::make_shared<gsl::sir::Field>(name, getLocation(member)));
    } else {
      GSL_ASSERT_MSG(0, "expected `clang::MemberExpr` in argument-list of stencil call");
    }
  }

  // Check if number of arguments match. Note that temporaries don't need to be provided, we
  // implicitly generate a new temporary for each temporary storage in the stencil
  std::size_t parsedArgs = fields.size();

  const auto& requiredFields = stencilIt->second->Fields;
  std::size_t requiredArgs =
      std::accumulate(requiredFields.begin(), requiredFields.end(), 0,
                      [](std::size_t acc, const std::shared_ptr<gsl::sir::Field>& field) {
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

  auto SIRStencilCall = std::make_shared<gsl::sir::StencilCall>(callee, getLocation(stencilCall));
  SIRStencilCall->Args = std::move(fields);

  GSL_LOG(INFO) << "Done parsing stencil call";
  return std::make_shared<gsl::StencilCallDeclStmt>(SIRStencilCall, SIRStencilCall->Loc);
}

std::shared_ptr<gsl::VerticalRegionDeclStmt>
StencilParser::parseVerticalRegion(clang::CXXForRangeStmt* verticalRegion) {
  using namespace clang;
  using namespace llvm;

  GSL_LOG(INFO) << "Parsing vertical region at " << getLocation(verticalRegion);

  // The idea is to translate each vertical region, given as a C++11 range-based for loop (i.e
  // `for(auto k : {X,X})`) into a VerticalRegionDeclStmt

  // Extract the Interval from the loop bounds
  IntervalResolver intervalResolver(this);
  intervalResolver.resolve(verticalRegion);

  // Extract the Do-Method body (AST) from the loop body
  auto SIRBlockStmt = std::make_shared<gsl::BlockStmt>(getLocation(verticalRegion));

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
      for(Stmt* stmt : compoundStmt->body())
        SIRBlockStmt->insert_back(
            stmtResolver.resolveStmt(stmt, ClangASTStmtResolver::AK_StencilBody));

    } else {
      // Case for(...) XXX
      SIRBlockStmt->insert_back(stmtResolver.resolveStmt(verticalRegion->getBody(),
                                                         ClangASTStmtResolver::AK_StencilBody));
    }
  }

  // Assemble the vertical region and register it within the current Stencil
  auto SIRAST = std::make_shared<gsl::AST>(std::move(SIRBlockStmt));

  auto intervalPair = intervalResolver.getInterval();

  auto SIRVerticalRegion = std::make_shared<gsl::sir::VerticalRegion>(
      SIRAST, intervalPair.first, intervalPair.second, getLocation(verticalRegion));

  GSL_LOG(INFO) << "Done parsing vertical region";
  return std::make_shared<gsl::VerticalRegionDeclStmt>(SIRVerticalRegion, SIRVerticalRegion->Loc);
}

std::shared_ptr<gsl::BoundaryConditionDeclStmt>
StencilParser::parseBoundaryCondition(clang::CXXConstructExpr* boundaryCondition) {
  using namespace clang;
  using namespace llvm;

  GSL_ASSERT(boundaryCondition->getConstructor()->getNameAsString() == "boundary_condition");

  GSL_LOG(INFO) << "Parsing boundary-condition at "
                << getFilename(
                       boundaryCondition->getLocation().printToString(context_->getSourceManager()))
                       .str()
                << " ...";

  BoundaryConditionResolver resolver(this);
  resolver.resolve(boundaryCondition);

  auto ASTBoundaryCondition = std::make_shared<gsl::BoundaryConditionDeclStmt>(
      resolver.getFunctor(), getLocation(boundaryCondition));

  for(const auto& name : resolver.getFields()) {
    gsl::sir::Stencil* curStencil = currentParserRecord_->CurrentStencil;

    auto it = std::find_if(
        curStencil->Fields.begin(), curStencil->Fields.end(),
        [&name](const std::shared_ptr<gsl::sir::Field>& field) { return field->Name == name; });

    if(it == curStencil->Fields.end()) {
      auto builder = reportDiagnostic(boundaryCondition->getLocation(),
                                      Diagnostics::DiagKind::err_boundary_condition_invalid_arg);
      builder << name;
    } else {
      ASTBoundaryCondition->getFields().push_back(*it);
    }
  }

  GSL_LOG(INFO) << "Done parsing boundary-condition";
  return ASTBoundaryCondition;
}

gsl::SourceLocation StencilParser::getLocation(clang::Decl* decl) const {
  clang::PresumedLoc ploc = context_->getSourceManager().getPresumedLoc(decl->getLocation());
  return gsl::SourceLocation(ploc.getLine(), ploc.getColumn());
}

gsl::SourceLocation StencilParser::getLocation(clang::Stmt* stmt) const {
  clang::PresumedLoc ploc = context_->getSourceManager().getPresumedLoc(stmt->getLocStart());
  return gsl::SourceLocation(ploc.getLine(), ploc.getColumn());
}

} // namespace gtclang
