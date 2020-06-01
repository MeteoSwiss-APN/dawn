//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStringifier.h"
#include "dawn/IIR/AccessUtils.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/Unreachable.h"

#include <numeric>
#include <optional>
#include <ostream>

namespace dawn {
namespace iir {

using ::dawn::operator<<;

StencilFunctionInstantiation::StencilFunctionInstantiation(
    StencilInstantiation* context, const std::shared_ptr<iir::StencilFunCallExpr>& expr,
    const std::shared_ptr<sir::StencilFunction>& function, const std::shared_ptr<iir::AST>& ast,
    const Interval& interval, bool isNested)
    : stencilInstantiation_(context), metadata_(context->getMetaData()), expr_(expr),
      function_(function), ast_(ast), interval_(interval), hasReturn_(false), isNested_(isNested),
      doMethod_(std::make_unique<DoMethod>(interval, context->getMetaData())) {
  DAWN_ASSERT(context);
  DAWN_ASSERT(function);
}

StencilFunctionInstantiation StencilFunctionInstantiation::clone() const {
  // The SIR object function_ is not cloned, but copied, since the SIR is considered immuatble
  StencilFunctionInstantiation stencilFun(
      stencilInstantiation_, std::static_pointer_cast<iir::StencilFunCallExpr>(expr_->clone()),
      function_, ast_->clone(), interval_, isNested_);

  stencilFun.hasReturn_ = hasReturn_;
  stencilFun.argsBound_ = argsBound_;
  stencilFun.ArgumentIndexToCallerAccessIDMap_ = ArgumentIndexToCallerAccessIDMap_;
  stencilFun.ArgumentIndexToStencilFunctionInstantiationMap_ =
      ArgumentIndexToStencilFunctionInstantiationMap_;
  stencilFun.ArgumentIndexToCallerDirectionMap_ = ArgumentIndexToCallerDirectionMap_;
  stencilFun.ArgumentIndexToCallerOffsetMap_ = ArgumentIndexToCallerOffsetMap_;
  stencilFun.CallerAccessIDToInitialOffsetMap_ = CallerAccessIDToInitialOffsetMap_;
  stencilFun.AccessIDToNameMap_ = AccessIDToNameMap_;
  stencilFun.LiteralAccessIDToNameMap_ = LiteralAccessIDToNameMap_;
  stencilFun.ExprToStencilFunctionInstantiationMap_ = ExprToStencilFunctionInstantiationMap_;
  stencilFun.calleeFields_ = calleeFields_;
  stencilFun.callerFields_ = callerFields_;
  stencilFun.unusedFields_ = unusedFields_;
  stencilFun.GlobalVariableAccessIDSet_ = GlobalVariableAccessIDSet_;

  stencilFun.doMethod_ = doMethod_->clone();

  return stencilFun;
}

ast::Offsets StencilFunctionInstantiation::evalOffsetOfFieldAccessExpr(
    const std::shared_ptr<iir::FieldAccessExpr>& expr, bool applyInitialOffset) const {

  // Get the offsets we know so far (i.e the constant offset)
  ast::Offsets offset = expr->getOffset();

  // Apply the initial offset (e.g if we call a function `avg(in(i+1))` we have to shift all
  // accesses of the field `in` by [1, 0, 0])
  if(applyInitialOffset) {
    const ast::Offsets& initialOffset = getCallerInitialOffsetFromAccessID(iir::getAccessID(expr));
    offset += initialOffset;
  }

  int sign = expr->negateOffset() ? -1 : 1;

  // Iterate the argument map (if index is *not* -1, we have to lookup the dimension or offset of
  // the directional or offset argument)
  for(int i = 0; i < expr->getArgumentMap().size(); ++i) {
    const int argIndex = expr->getArgumentMap()[i];

    if(argIndex != -1) {
      const int argOffset = expr->getArgumentOffset()[i];

      // Resolve the directions and offsets
      // Note: Offsets could implement directions, then this could be simplified because we just
      // want to do offset[my_direction] += sign * argOffset;
      std::array<int, 3> addOffset{};
      if(isArgDirection(argIndex))
        addOffset[getCallerDimensionOfArgDirection(argIndex)] += sign * argOffset;
      else {
        const auto& instantiatedOffset = getCallerOffsetOfArgOffset(argIndex);
        addOffset[instantiatedOffset[0]] = sign * (argOffset + instantiatedOffset[1]);
      }
      offset += ast::Offsets{ast::cartesian, addOffset};
    }
  }

  return offset;
}

std::vector<std::shared_ptr<sir::StencilFunctionArg>>&
StencilFunctionInstantiation::getArguments() {
  return function_->Args;
}

const std::vector<std::shared_ptr<sir::StencilFunctionArg>>&
StencilFunctionInstantiation::getArguments() const {
  return function_->Args;
}

//===------------------------------------------------------------------------------------------===//
//     Argument Maps
//===------------------------------------------------------------------------------------------===//

int StencilFunctionInstantiation::getCallerDimensionOfArgDirection(int argumentIndex) const {
  DAWN_ASSERT(ArgumentIndexToCallerDirectionMap_.count(argumentIndex));
  return ArgumentIndexToCallerDirectionMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setCallerDimensionOfArgDirection(int argumentIndex,
                                                                    int dimension) {
  ArgumentIndexToCallerDirectionMap_[argumentIndex] = dimension;
}

bool StencilFunctionInstantiation::isArgBoundAsOffset(int argumentIndex) const {
  return ArgumentIndexToCallerOffsetMap_.count(argumentIndex);
}

bool StencilFunctionInstantiation::isArgBoundAsDirection(int argumentIndex) const {
  return ArgumentIndexToCallerDirectionMap_.count(argumentIndex);
}

bool StencilFunctionInstantiation::isArgBoundAsFunctionInstantiation(int argumentIndex) const {
  return ArgumentIndexToStencilFunctionInstantiationMap_.count(argumentIndex);
}

bool StencilFunctionInstantiation::isArgBoundAsFieldAccess(int argumentIndex) const {
  return ArgumentIndexToCallerAccessIDMap_.count(argumentIndex);
}

const Array2i& StencilFunctionInstantiation::getCallerOffsetOfArgOffset(int argumentIndex) const {
  DAWN_ASSERT(ArgumentIndexToCallerOffsetMap_.count(argumentIndex));
  return ArgumentIndexToCallerOffsetMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setCallerOffsetOfArgOffset(int argumentIndex,
                                                              const Array2i& offset) {
  ArgumentIndexToCallerOffsetMap_[argumentIndex] = offset;
}

int StencilFunctionInstantiation::getCallerAccessIDOfArgField(int argumentIndex) const {
  return ArgumentIndexToCallerAccessIDMap_.at(argumentIndex);
}

void StencilFunctionInstantiation::setCallerAccessIDOfArgField(int argumentIndex,
                                                               int callerAccessID) {
  ArgumentIndexToCallerAccessIDMap_[argumentIndex] = callerAccessID;
}

std::shared_ptr<StencilFunctionInstantiation>
StencilFunctionInstantiation::getFunctionInstantiationOfArgField(int argumentIndex) const {
  DAWN_ASSERT(ArgumentIndexToStencilFunctionInstantiationMap_.count(argumentIndex));
  return ArgumentIndexToStencilFunctionInstantiationMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setFunctionInstantiationOfArgField(
    int argumentIndex, const std::shared_ptr<StencilFunctionInstantiation>& func) {
  ArgumentIndexToStencilFunctionInstantiationMap_[argumentIndex] = func;
}

const ast::Offsets&
StencilFunctionInstantiation::getCallerInitialOffsetFromAccessID(int callerAccessID) const {
  DAWN_ASSERT(CallerAccessIDToInitialOffsetMap_.count(callerAccessID));
  return CallerAccessIDToInitialOffsetMap_.find(callerAccessID)->second;
}

void StencilFunctionInstantiation::setCallerInitialOffsetFromAccessID(int callerAccessID,
                                                                      const ast::Offsets& offset) {
  CallerAccessIDToInitialOffsetMap_.emplace(callerAccessID, offset);
}

bool StencilFunctionInstantiation::isProvidedByStencilFunctionCall(int callerAccessID) const {
  auto pos = std::find_if(ArgumentIndexToCallerAccessIDMap_.begin(),
                          ArgumentIndexToCallerAccessIDMap_.end(),
                          [&](const std::pair<int, int>& p) { return p.second == callerAccessID; });

  // accessID is not an argument to stencil function
  if(pos == ArgumentIndexToCallerAccessIDMap_.end())
    return false;
  return isArgStencilFunctionInstantiation(pos->first);
}

int StencilFunctionInstantiation::getArgumentIndexFromCallerAccessID(int callerAccessID) const {
  for(std::size_t argIdx = 0; argIdx < function_->Args.size(); ++argIdx)
    if(isArgField(argIdx) || isArgStencilFunctionInstantiation(callerAccessID))
      if(getCallerAccessIDOfArgField(argIdx) == callerAccessID)
        return argIdx;
  dawn_unreachable("invalid AccessID");
}

const std::string&
StencilFunctionInstantiation::getOriginalNameFromCallerAccessID(int callerAccessID) const {
  for(std::size_t argIdx = 0; argIdx < function_->Args.size(); ++argIdx) {
    if(sir::Field* field = dyn_cast<sir::Field>(function_->Args[argIdx].get())) {
      if(getCallerAccessIDOfArgField(argIdx) == callerAccessID)
        return field->Name;
    }
  }
  dawn_unreachable("invalid AccessID");
}

const Field&
StencilFunctionInstantiation::getCallerFieldFromArgumentIndex(int argumentIndex) const {
  int callerAccessID = getCallerAccessIDOfArgField(argumentIndex);

  for(const Field& field : callerFields_)
    if(field.getAccessID() == callerAccessID)
      return field;

  dawn_unreachable("invalid argument index of field");
}

const std::vector<Field>& StencilFunctionInstantiation::getCallerFields() const {
  return callerFields_;
}

const std::vector<Field>& StencilFunctionInstantiation::getCalleeFields() const {
  return calleeFields_;
}

bool StencilFunctionInstantiation::isArgOffset(int argumentIndex) const {
  return isa<sir::Offset>(function_->Args[argumentIndex].get());
}

bool StencilFunctionInstantiation::isArgDirection(int argumentIndex) const {
  return isa<sir::Direction>(function_->Args[argumentIndex].get());
}

bool StencilFunctionInstantiation::isArgField(int argumentIndex) const {
  return isa<sir::Field>(function_->Args[argumentIndex].get());
}

bool StencilFunctionInstantiation::isArgStencilFunctionInstantiation(int argumentIndex) const {
  return ArgumentIndexToStencilFunctionInstantiationMap_.count(argumentIndex);
}

//===----------------------------------------------------------------------------------------===//
//     Expr/Stmt to Caller AccessID Maps
//===----------------------------------------------------------------------------------------===//

std::string StencilFunctionInstantiation::getFieldNameFromAccessID(int AccessID) const {
  // As we store the caller accessIDs, we have to get the name of the field from the context!
  // TODO have a check for what is a literal range
  if(AccessID < 0)
    return getNameFromLiteralAccessID(AccessID);
  else if(metadata_.isAccessType(FieldAccessType::Field, AccessID) ||
          metadata_.isAccessType(iir::FieldAccessType::GlobalVariable, AccessID))
    return metadata_.getFieldNameFromAccessID(AccessID);
  else {
    DAWN_ASSERT(AccessIDToNameMap_.count(AccessID));
    return AccessIDToNameMap_.find(AccessID)->second;
  }
}

void StencilFunctionInstantiation::setAccessIDOfGlobalVariable(int AccessID) {
  //  setAccessIDNamePair(AccessID, name);
  GlobalVariableAccessIDSet_.insert(AccessID);
}

const std::string& StencilFunctionInstantiation::getNameFromLiteralAccessID(int AccessID) const {
  auto it = LiteralAccessIDToNameMap_.find(AccessID);
  DAWN_ASSERT_MSG(it != LiteralAccessIDToNameMap_.end(), "Invalid Literal");
  return it->second;
}

std::string StencilFunctionInstantiation::getNameFromAccessID(int accessID) const {
  if(isLiteral(accessID)) {
    return getNameFromLiteralAccessID(accessID);
  } else if(metadata_.isAccessType(FieldAccessType::Field, accessID) ||
            isProvidedByStencilFunctionCall(accessID)) {
    return getOriginalNameFromCallerAccessID(accessID);
  } else {
    return getFieldNameFromAccessID(accessID);
  }
}

std::unordered_map<int, std::string>& StencilFunctionInstantiation::getLiteralAccessIDToNameMap() {
  return LiteralAccessIDToNameMap_;
}
const std::unordered_map<int, std::string>&
StencilFunctionInstantiation::getLiteralAccessIDToNameMap() const {
  return LiteralAccessIDToNameMap_;
}

std::unordered_map<int, std::string>& StencilFunctionInstantiation::getAccessIDToNameMap() {
  return AccessIDToNameMap_;
}

const std::unordered_map<int, std::string>&
StencilFunctionInstantiation::getAccessIDToNameMap() const {
  return AccessIDToNameMap_;
}
std::unordered_map<int, ast::Offsets>&
StencilFunctionInstantiation::getCallerAccessIDToInitialOffsetMap() {
  return CallerAccessIDToInitialOffsetMap_;
}
const std::unordered_map<int, ast::Offsets>&
StencilFunctionInstantiation::getCallerAccessIDToInitialOffsetMap() const {
  return CallerAccessIDToInitialOffsetMap_;
}

const std::unordered_map<std::shared_ptr<iir::StencilFunCallExpr>,
                         std::shared_ptr<StencilFunctionInstantiation>>&
StencilFunctionInstantiation::getExprToStencilFunctionInstantiationMap() const {
  return ExprToStencilFunctionInstantiationMap_;
}

void StencilFunctionInstantiation::insertExprToStencilFunction(
    const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
  DAWN_ASSERT(!ExprToStencilFunctionInstantiationMap_.count(stencilFun->getExpression()));

  ExprToStencilFunctionInstantiationMap_.emplace(stencilFun->getExpression(), stencilFun);
}

void StencilFunctionInstantiation::removeStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) {
  ExprToStencilFunctionInstantiationMap_.erase(expr);
}

std::shared_ptr<StencilFunctionInstantiation>
StencilFunctionInstantiation::getStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

bool StencilFunctionInstantiation::hasStencilFunctionInstantiation(
    const std::shared_ptr<iir::StencilFunCallExpr>& expr) const {
  return (ExprToStencilFunctionInstantiationMap_.find(expr) !=
          ExprToStencilFunctionInstantiationMap_.end());
}

const std::vector<std::shared_ptr<iir::Stmt>>& StencilFunctionInstantiation::getStatements() const {
  return doMethod_->getAST().getStatements();
}

//===------------------------------------------------------------------------------------------===//
//     Accesses & Fields
//===------------------------------------------------------------------------------------------===//

void StencilFunctionInstantiation::update() {
  callerFields_.clear();
  calleeFields_.clear();
  unusedFields_.clear();

  // Compute the fields and their intended usage. Fields can be in one of three states: `Output`,
  // `InputOutput` or `Input` which implements the following state machine:
  //
  //    +-------+                               +--------+
  //    | Input |                               | Output |
  //    +-------+                               +--------+
  //        |                                       |
  //        |            +-------------+            |
  //        +----------> | InputOutput | <----------+
  //                     +-------------+
  //
  std::unordered_map<int, Field> inputOutputFields;
  std::unordered_map<int, Field> inputFields;
  std::unordered_map<int, Field> outputFields;

  for(const auto& stmt : doMethod_->getAST().getStatements()) {
    const auto& access = stmt->getData<IIRStmtData>().CallerAccesses;
    DAWN_ASSERT(access);

    for(const auto& accessPair : access->getWriteAccesses()) {
      int AccessID = accessPair.first;

      // Does this AccessID correspond to a field access?
      if(!isProvidedByStencilFunctionCall(AccessID) &&
         !metadata_.isAccessType(FieldAccessType::Field, AccessID))
        continue;
      auto&& dims = metadata_.isAccessType(FieldAccessType::Field, AccessID)
                        ? metadata_.getFieldDimensions(AccessID)
                        : sir::FieldDimensions(
                              sir::HorizontalFieldDimension(ast::cartesian, {true, true}),
                              true); // TODO sparse_dim: this is a hack. Ideally we don't want
                                     // to create Field when the argument is a function call.
      AccessUtils::recordWriteAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                     std::optional<Extents>(), interval_, std::move(dims));
    }

    for(const auto& accessPair : access->getReadAccesses()) {
      int AccessID = accessPair.first;

      // Does this AccessID correspond to a field access?
      if(!isProvidedByStencilFunctionCall(AccessID) &&
         !metadata_.isAccessType(FieldAccessType::Field, AccessID))
        continue;

      auto&& dims = metadata_.isAccessType(FieldAccessType::Field, AccessID)
                        ? metadata_.getFieldDimensions(AccessID)
                        : sir::FieldDimensions(
                              sir::HorizontalFieldDimension(ast::cartesian, {true, true}),
                              true); // TODO sparse_dim: this is a hack. Ideally we don't want
                                     // to create Field when the argument is a function call.
      AccessUtils::recordReadAccess(inputOutputFields, inputFields, outputFields, AccessID,
                                    std::optional<Extents>(), interval_, std::move(dims));
    }
  }

  // Add AccessIDs of unused fields i.e fields which are passed as arguments but never referenced.
  for(const auto& argIdxCallerAccessIDPair : ArgumentIndexToCallerAccessIDMap_) {
    int AccessID = argIdxCallerAccessIDPair.second;
    if(!inputFields.count(AccessID) && !outputFields.count(AccessID) &&
       !inputOutputFields.count(AccessID)) {
      auto&& dims = metadata_.isAccessType(FieldAccessType::Field, AccessID)
                        ? metadata_.getFieldDimensions(AccessID)
                        : sir::FieldDimensions(
                              sir::HorizontalFieldDimension(ast::cartesian, {true, true}),
                              true); // TODO sparse_dim: this is a hack. Ideally we don't want
                                     // to create Field when the argument is a function call.
      inputFields.emplace(AccessID, Field(AccessID, Field::IntendKind::Input, Extents{}, Extents{},
                                          interval_, std::move(dims)));
      unusedFields_.insert(AccessID);
    }
  }

  std::vector<Field> calleeFieldsUnordered;
  std::vector<Field> callerFieldsUnordered;

  // Merge inputFields, outputFields and fields. Note that caller and callee fields are the same,
  // the only difference is that in the caller fields we apply the inital offset to the extents
  // while in the callee fields we do not.
  for(auto AccessIDFieldPair : outputFields) {
    calleeFieldsUnordered.push_back(AccessIDFieldPair.second);
    callerFieldsUnordered.push_back(AccessIDFieldPair.second);
  }

  for(auto AccessIDFieldPair : inputOutputFields) {
    calleeFieldsUnordered.push_back(AccessIDFieldPair.second);
    callerFieldsUnordered.push_back(AccessIDFieldPair.second);
  }

  for(auto AccessIDFieldPair : inputFields) {
    calleeFieldsUnordered.push_back(AccessIDFieldPair.second);
    callerFieldsUnordered.push_back(AccessIDFieldPair.second);
  }

  if(calleeFieldsUnordered.empty() || callerFieldsUnordered.empty()) {
    DAWN_LOG(WARNING) << "no fields referenced in this stencil function";
  } else {

    // Accumulate the extent of the fields (note that here the callee and caller fields differ
    // as the caller fields have the *initial* extent (e.g in `avg(u(i+1))` u has an initial extent
    // of [1, 0, 0])
    auto computeAccesses = [&](std::vector<Field>& fields, bool callerAccesses) {
      // Index to speedup lookup into fields map
      std::unordered_map<int, std::vector<Field>::iterator> AccessIDToFieldMap;
      for(auto it = fields.begin(), end = fields.end(); it != end; ++it)
        AccessIDToFieldMap.insert(std::make_pair(it->getAccessID(), it));

      // Accumulate the extents of each field in this stage
      for(const auto& stmt : doMethod_->getAST().getStatements()) {
        const auto& access = callerAccesses ? stmt->getData<IIRStmtData>().CallerAccesses
                                            : stmt->getData<IIRStmtData>().CalleeAccesses;

        // first => AccessID, second => Extent
        for(auto& accessPair : access->getWriteAccesses()) {
          if(!isProvidedByStencilFunctionCall(accessPair.first) &&
             !metadata_.isAccessType(FieldAccessType::Field, accessPair.first))
            continue;

          AccessIDToFieldMap[accessPair.first]->mergeWriteExtents(accessPair.second);
        }

        for(const auto& accessPair : access->getReadAccesses()) {
          if(!isProvidedByStencilFunctionCall(accessPair.first) &&
             !metadata_.isAccessType(FieldAccessType::Field, accessPair.first))
            continue;

          AccessIDToFieldMap[accessPair.first]->mergeReadExtents(accessPair.second);
        }
      }
    };

    computeAccesses(callerFieldsUnordered, true);
    computeAccesses(calleeFieldsUnordered, false);
  }

  // Reorder the fields s.t they match the order in which they were decalred in the stencil-function
  for(int argIdx = 0; argIdx < getArguments().size(); ++argIdx) {
    if(isArgField(argIdx)) {
      int AccessID = getCallerAccessIDOfArgField(argIdx);

      auto insertField = [&](std::vector<Field>& fieldsOrdered,
                             std::vector<Field>& fieldsUnordered) {
        auto it = std::find_if(fieldsUnordered.begin(), fieldsUnordered.end(),
                               [&](const Field& field) { return field.getAccessID() == AccessID; });
        DAWN_ASSERT(it != fieldsUnordered.end());
        fieldsOrdered.push_back(*it);
      };

      insertField(callerFields_, callerFieldsUnordered);
      insertField(calleeFields_, calleeFieldsUnordered);
    }
  }
}

bool StencilFunctionInstantiation::isFieldUnused(int AccessID) const {
  return unusedFields_.count(AccessID);
}

//===------------------------------------------------------------------------------------------===//
//     Miscellaneous
//===------------------------------------------------------------------------------------------===//

std::string
StencilFunctionInstantiation::makeCodeGenName(const StencilFunctionInstantiation& stencilFun) {
  std::string name = stencilFun.getName();

  for(std::size_t argIdx = 0; argIdx < stencilFun.getStencilFunction()->Args.size(); ++argIdx) {
    if(stencilFun.isArgOffset(argIdx)) {
      const auto& offset = stencilFun.getCallerOffsetOfArgOffset(argIdx);
      name += "_" + dim2str(offset[0]) + "_";
      if(offset[1] != 0)
        name += (offset[1] > 0 ? "plus_" : "minus_");
      name += std::to_string(std::abs(offset[1]));

    } else if(stencilFun.isArgDirection(argIdx)) {
      name += "_" + dim2str(stencilFun.getCallerDimensionOfArgDirection(argIdx));
    } else if(stencilFun.isArgStencilFunctionInstantiation(argIdx)) {
      StencilFunctionInstantiation& argFunction =
          *(stencilFun.getFunctionInstantiationOfArgField(argIdx));
      name += "_" + makeCodeGenName(argFunction);
    }
  }

  name += "_" + Interval::makeCodeGenName(stencilFun.getInterval());
  return name;
}

void StencilFunctionInstantiation::setReturn(bool hasReturn) { hasReturn_ = hasReturn; }

bool StencilFunctionInstantiation::hasReturn() const { return hasReturn_; }

bool StencilFunctionInstantiation::isNested() const { return isNested_; }

size_t StencilFunctionInstantiation::numArgs() const { return function_->Args.size(); }

std::string StencilFunctionInstantiation::getArgNameFromFunctionCall(std::string fnCallName) const {

  for(std::size_t argIdx = 0; argIdx < numArgs(); ++argIdx) {
    if(!isArgField(argIdx) || !isArgStencilFunctionInstantiation(argIdx))
      continue;

    if(fnCallName == getFunctionInstantiationOfArgField(argIdx)->getName()) {
      sir::Field* field = dyn_cast<sir::Field>(function_->Args[argIdx].get());
      return field->Name;
    }
  }
  DAWN_ASSERT_MSG(0, "arg field of callee being a stencial function at caller not found");
  return "";
}

void StencilFunctionInstantiation::dump(std::ostream& os) const {
  os << "\nStencilFunction : " << getName() << " " << getInterval() << "\n";
  os << MakeIndent<1>::value << "Arguments:\n";

  for(std::size_t argIdx = 0; argIdx < numArgs(); ++argIdx) {

    os << MakeIndent<2>::value << "arg(" << argIdx << ") : ";

    if(isArgOffset(argIdx)) {
      int dim = getCallerOffsetOfArgOffset(argIdx)[0];
      int offset = getCallerOffsetOfArgOffset(argIdx)[1];
      os << "Offset : " << dim2str(dim);
      if(offset != 0)
        os << (offset > 0 ? "+" : "") << offset;
    } else if(isArgField(argIdx)) {
      sir::Field* field = dyn_cast<sir::Field>(function_->Args[argIdx].get());
      os << "Field : " << field->Name << " -> ";
      if(isArgStencilFunctionInstantiation(argIdx)) {
        os << "stencil-function-call:" << getFunctionInstantiationOfArgField(argIdx)->getName();
      } else {
        int callerAccessID = getCallerAccessIDOfArgField(argIdx);
        os << metadata_.getFieldNameFromAccessID(callerAccessID) << "  "
           << to_string(getCallerInitialOffsetFromAccessID(callerAccessID));
      }

    } else {
      os << "Direction : " << dim2str(getCallerDimensionOfArgDirection(argIdx));
    }
    os << "\n";
  }

  os << MakeIndent<1>::value << "Accesses (including initial offset):\n";

  const auto& statements = getAST()->getRoot()->getStatements();
  for(std::size_t i = 0; i < statements.size(); ++i) {
    os << "\033[1m" << iir::ASTStringifier::toString(statements[i], 2 * DAWN_PRINT_INDENT)
       << "\033[0m";
    const auto& callerAccesses =
        doMethod_->getAST().getStatements()[i]->getData<IIRStmtData>().CallerAccesses;
    if(callerAccesses)
      os << callerAccesses->toString(
                [&](int AccessID) { return this->getNameFromAccessID(AccessID); },
                3 * DAWN_PRINT_INDENT)
         << "\n";
  }
}

void StencilFunctionInstantiation::closeFunctionBindings() {
  std::vector<int> arglist(getArguments().size());
  std::iota(arglist.begin(), arglist.end(), 0);

  closeFunctionBindings(arglist);
}
void StencilFunctionInstantiation::closeFunctionBindings(const std::vector<int>& arglist) {
  // finalize the bindings of some of the arguments that are not yet instantiated
  const auto& arguments = getArguments();

  for(int argIdx : arglist) {
    if(isa<sir::Field>(*arguments[argIdx])) {
      if(isArgStencilFunctionInstantiation(argIdx)) {

        // The field is provided by a stencil function call, we create a new AccessID for this
        // "temporary" field
        int AccessID = stencilInstantiation_->nextUID();

        setCallerAccessIDOfArgField(argIdx, AccessID);
        setCallerInitialOffsetFromAccessID(AccessID, ast::Offsets{ast::cartesian});
      }
    }
  }

  argsBound_ = true;
}

void StencilFunctionInstantiation::checkFunctionBindings() const {

  const auto& arguments = getArguments();

  for(std::size_t argIdx = 0; argIdx < arguments.size(); ++argIdx) {
    // check that all arguments of all possible types are assigned
    if(isa<sir::Field>(*arguments[argIdx])) {
      DAWN_ASSERT_MSG(
          (isArgBoundAsFieldAccess(argIdx) || isArgBoundAsFunctionInstantiation(argIdx)),
          std::string("Field access arg not bound for function " + function_->Name).c_str());
    } else if(isa<sir::Direction>(*arguments[argIdx])) {
      DAWN_ASSERT_MSG(
          (isArgBoundAsDirection(argIdx)),
          std::string("Direction arg not bound for function " + function_->Name).c_str());
    } else if(isa<sir::Offset>(*arguments[argIdx])) {
      DAWN_ASSERT_MSG((isArgBoundAsOffset(argIdx)),
                      std::string("Offset arg not bound for function " + function_->Name).c_str());
    } else
      dawn_unreachable("Argument not supported");
  }

  // check that the list of <statement,access> are set for all statements
  DAWN_ASSERT_MSG(
      (getAST()->getRoot()->getStatements().size() == doMethod_->getAST().getStatements().size()),
      "AST has different number of statements with respect to DoMethod's AST");
}

} // namespace iir
} // namespace dawn
