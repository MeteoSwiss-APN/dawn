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

#include "dawn/Optimizer/StencilFunctionInstantiation.h"
#include "dawn/Optimizer/Renaming.h"
#include "dawn/Optimizer/StencilInstantiation.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/Unreachable.h"
#include <iostream>

namespace dawn {

static std::string dim2str(int dim) {
  switch(dim) {
  case 0:
    return "i";
  case 1:
    return "j";
  case 2:
    return "k";
  default:
    dawn_unreachable("invalid dimension");
  }
};

StencilFunctionInstantiation::StencilFunctionInstantiation(
    StencilInstantiation* context, const std::shared_ptr<StencilFunCallExpr>& expr,
    sir::StencilFunction* function, const std::shared_ptr<AST>& ast, const Interval& interval,
    bool isNested)
    : stencilInstantiation_(context), expr_(expr), function_(function), ast_(ast),
      interval_(interval), hasReturn_(false), isNested_(isNested) {}

Array3i StencilFunctionInstantiation::evalOffsetOfFieldAccessExpr(
    const std::shared_ptr<FieldAccessExpr>& expr, bool applyInitialOffset) const {

  // Get the offsets we know so far (i.e the constant offset)
  Array3i offset = expr->getOffset();

  // Apply the initial offset (e.g if we call a function `avg(in(i+1))` we have to shift all
  // accesses of the field `in` by [1, 0, 0])
  if(applyInitialOffset) {
    const Array3i& initialOffset = getCallerInitialOffsetFromAccessID(getAccessIDFromExpr(expr));
    offset[0] += initialOffset[0];
    offset[1] += initialOffset[1];
    offset[2] += initialOffset[2];
  }

  int sign = expr->negateOffset() ? -1 : 1;

  // Iterate the argument map (if index is *not* -1, we have to lookup the dimension or offset of
  // the directional or offset argument)
  for(int i = 0; i < expr->getArgumentMap().size(); ++i) {
    const int argIndex = expr->getArgumentMap()[i];

    if(argIndex != -1) {
      const int argOffset = expr->getArgumentOffset()[i];

      // Resolve the directions and offsets
      if(isArgDirection(argIndex))
        offset[getCallerDimensionOfArgDirection(argIndex)] += sign * argOffset;
      else {
        const auto& instantiatedOffset = getCallerOffsetOfArgOffset(argIndex);
        offset[instantiatedOffset[0]] += sign * (argOffset + instantiatedOffset[1]);
      }
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
  return ArgumentIndexToCallerDirectionMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setCallerDimensionOfArgDirection(int argumentIndex,
                                                                    int dimension) {
  ArgumentIndexToCallerDirectionMap_[argumentIndex] = dimension;
}

const Array2i& StencilFunctionInstantiation::getCallerOffsetOfArgOffset(int argumentIndex) const {
  return ArgumentIndexToCallerOffsetMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setCallerOffsetOfArgOffset(int argumentIndex,
                                                              const Array2i& offset) {
  ArgumentIndexToCallerOffsetMap_[argumentIndex] = offset;
}

int StencilFunctionInstantiation::getCallerAccessIDOfArgField(int argumentIndex) const {
  return ArgumentIndexToCallerAccessIDMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setCallerAccessIDOfArgField(int argumentIndex,
                                                               int callerAccessID) {
  ArgumentIndexToCallerAccessIDMap_[argumentIndex] = callerAccessID;
}

StencilFunctionInstantiation*
StencilFunctionInstantiation::getFunctionInstantiationOfArgField(int argumentIndex) const {
  return ArgumentIndexToStencilFunctionInstantiationMap_.find(argumentIndex)->second;
}

void StencilFunctionInstantiation::setFunctionInstantiationOfArgField(
    int argumentIndex, StencilFunctionInstantiation* func) {
  ArgumentIndexToStencilFunctionInstantiationMap_[argumentIndex] = func;
}

const Array3i&
StencilFunctionInstantiation::getCallerInitialOffsetFromAccessID(int callerAccessID) const {
  return CallerAcceessIDToInitialOffsetMap_.find(callerAccessID)->second;
}

void StencilFunctionInstantiation::setCallerInitialOffsetFromAccessID(int callerAccessID,
                                                                      const Array3i& offset) {
  CallerAcceessIDToInitialOffsetMap_[callerAccessID] = offset;
}

bool StencilFunctionInstantiation::isProvidedByStencilFunctionCall(int callerAccessID) const {
  return isProvidedByStencilFunctionCall_.count(callerAccessID);
}

void StencilFunctionInstantiation::setIsProvidedByStencilFunctionCall(int callerAccessID) {
  isProvidedByStencilFunctionCall_.insert(callerAccessID);
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
  for(std::size_t argIdx = 0; argIdx < function_->Args.size(); ++argIdx)
    if(sir::Field* field = dyn_cast<sir::Field>(function_->Args[argIdx].get()))
      if(getCallerAccessIDOfArgField(argIdx) == callerAccessID)
        return field->Name;
  dawn_unreachable("invalid AccessID");
}

const Field&
StencilFunctionInstantiation::getCallerFieldFromArgumentIndex(int argumentIndex) const {
  int callerAccessID = getCallerAccessIDOfArgField(argumentIndex);

  for(const Field& field : callerFields_)
    if(field.AccessID == callerAccessID)
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

template <class MapType, class KeyType>
static void replaceKeyInMap(MapType& map, KeyType oldKey, KeyType newKey) {
  auto it = map.find(oldKey);
  if(it != map.end()) {
    std::swap(map[newKey], it->second);
    map.erase(it);
  }
}

void StencilFunctionInstantiation::renameCallerAccessID(int oldAccessID, int newAccessID) {
  // Update argument maps
  for(auto& argumentAccessIDPair : ArgumentIndexToCallerAccessIDMap_) {
    int& AccessID = argumentAccessIDPair.second;
    if(AccessID == oldAccessID)
      AccessID = newAccessID;
  }
  replaceKeyInMap(CallerAcceessIDToInitialOffsetMap_, oldAccessID, newAccessID);

  // Update AccessID to name map
  replaceKeyInMap(AccessIDToNameMap_, oldAccessID, newAccessID);

  // Update statements
  renameAccessIDInStmts(this, oldAccessID, newAccessID, statementAccessesPairs_);

  // Update accesses
  renameAccessIDInAccesses(this, oldAccessID, newAccessID, statementAccessesPairs_);

  // Recompute the fields
  update();
}

//===----------------------------------------------------------------------------------------===//
//     Expr/Stmt to Caller AccessID Maps
//===----------------------------------------------------------------------------------------===//

const std::string& StencilFunctionInstantiation::getNameFromAccessID(int AccessID) const {
  // As we store the caller accessIDs, we have to get the name of the field from the context!
  if(AccessID < 0)
    return getNameFromLiteralAccessID(AccessID);
  else if(stencilInstantiation_->isField(AccessID))
    return stencilInstantiation_->getNameFromAccessID(AccessID);
  else
    return AccessIDToNameMap_.find(AccessID)->second;
}

const std::string& StencilFunctionInstantiation::getNameFromLiteralAccessID(int AccessID) const {
  auto it = LiteralAccessIDToNameMap_.find(AccessID);
  DAWN_ASSERT_MSG(it != LiteralAccessIDToNameMap_.end(), "Invalid Literal");
  return it->second;
}

int StencilFunctionInstantiation::getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const {
  auto it = ExprToCallerAccessIDMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToCallerAccessIDMap_.end(), "Invalid Expr");
  return it->second;
}

int StencilFunctionInstantiation::getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const {
  auto it = StmtToCallerAccessIDMap_.find(stmt);
  DAWN_ASSERT_MSG(it != StmtToCallerAccessIDMap_.end(), "Invalid Stmt");
  return it->second;
}

const std::unordered_map<std::shared_ptr<Expr>, int>&
StencilFunctionInstantiation::getExprToCallerAccessIDMap() const {
  return ExprToCallerAccessIDMap_;
}

std::unordered_map<std::shared_ptr<Expr>, int>&
StencilFunctionInstantiation::getExprToCallerAccessIDMap() {
  return ExprToCallerAccessIDMap_;
}

const std::unordered_map<std::shared_ptr<Stmt>, int>&
StencilFunctionInstantiation::getStmtToCallerAccessIDMap() const {
  return StmtToCallerAccessIDMap_;
}

std::unordered_map<std::shared_ptr<Stmt>, int>&
StencilFunctionInstantiation::getStmtToCallerAccessIDMap() {
  return StmtToCallerAccessIDMap_;
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

std::unordered_map<std::shared_ptr<StencilFunCallExpr>, StencilFunctionInstantiation*>&
StencilFunctionInstantiation::getExprToStencilFunctionInstantiationMap() {
  return ExprToStencilFunctionInstantiationMap_;
}

const std::unordered_map<std::shared_ptr<StencilFunCallExpr>, StencilFunctionInstantiation*>&
StencilFunctionInstantiation::getExprToStencilFunctionInstantiationMap() const {
  return ExprToStencilFunctionInstantiationMap_;
}

StencilFunctionInstantiation* StencilFunctionInstantiation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

const StencilFunctionInstantiation* StencilFunctionInstantiation::getStencilFunctionInstantiation(
    const std::shared_ptr<StencilFunCallExpr>& expr) const {
  auto it = ExprToStencilFunctionInstantiationMap_.find(expr);
  DAWN_ASSERT_MSG(it != ExprToStencilFunctionInstantiationMap_.end(), "Invalid stencil function");
  return it->second;
}

std::vector<std::shared_ptr<StatementAccessesPair>>&
StencilFunctionInstantiation::getStatementAccessesPairs() {
  return statementAccessesPairs_;
}

const std::vector<std::shared_ptr<StatementAccessesPair>>&
StencilFunctionInstantiation::getStatementAccessesPairs() const {
  return statementAccessesPairs_;
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
  std::set<int> inputOutputFields;
  std::set<int> inputFields;
  std::set<int> outputFields;

  for(const auto& statementAccessesPair : statementAccessesPairs_) {
    auto& access = statementAccessesPair->getAccesses();

    for(const auto& accessPair : access->getWriteAccesses()) {
      int AccessID = accessPair.first;

      // Does this AccessID correspond to a field access?
      if(!isProvidedByStencilFunctionCall(AccessID) && !stencilInstantiation_->isField(AccessID))
        continue;

      // Field was recorded as `InputOutput`, state can't change ...
      if(inputOutputFields.count(AccessID))
        continue;

      // Field was recorded as `Input`, change it's state to `InputOutput`
      if(inputFields.count(AccessID)) {
        inputOutputFields.insert(AccessID);
        inputFields.erase(AccessID);
        continue;
      }

      // Field not yet present, record it as output
      outputFields.insert(AccessID);
    }

    for(const auto& accessPair : access->getReadAccesses()) {
      int AccessID = accessPair.first;

      // Does this AccessID correspond to a field access?
      if(!isProvidedByStencilFunctionCall(AccessID) && !stencilInstantiation_->isField(AccessID))
        continue;

      // Field was recorded as `InputOutput`, state can't change ...
      if(inputOutputFields.count(AccessID))
        continue;

      // Field was recorded as `Output`, change it's state to `InputOutput`
      if(outputFields.count(AccessID)) {
        inputOutputFields.insert(AccessID);
        outputFields.erase(AccessID);
        continue;
      }

      // Field not yet present, record it as input
      inputFields.insert(AccessID);
    }
  }

  // Add AccessIDs of unused fields i.e fields which are passed as arguments but never referenced.
  for(const auto& argIdxCallerAccessIDPair : ArgumentIndexToCallerAccessIDMap_) {
    int AccessID = argIdxCallerAccessIDPair.second;
    if(!inputFields.count(AccessID) && !outputFields.count(AccessID) &&
       !inputOutputFields.count(AccessID)) {
      inputFields.insert(AccessID);
      unusedFields_.insert(AccessID);
    }
  }

  std::vector<Field> calleeFieldsUnordered;
  std::vector<Field> callerFieldsUnordered;

  // Merge inputFields, outputFields and fields. Note that caller and callee fields are the same,
  // the only difference is that in the caller fields we apply the inital offset to the extents
  // while in the callee fields we do not.
  for(int AccessID : outputFields) {
    calleeFieldsUnordered.emplace_back(AccessID, Field::IK_Output);
    callerFieldsUnordered.emplace_back(AccessID, Field::IK_Output);
  }

  for(int AccessID : inputOutputFields) {
    calleeFieldsUnordered.emplace_back(AccessID, Field::IK_InputOutput);
    callerFieldsUnordered.emplace_back(AccessID, Field::IK_InputOutput);
  }

  for(int AccessID : inputFields) {
    calleeFieldsUnordered.emplace_back(AccessID, Field::IK_Input);
    callerFieldsUnordered.emplace_back(AccessID, Field::IK_Input);
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
        AccessIDToFieldMap.insert(std::make_pair(it->AccessID, it));

      // Accumulate the extents of each field in this stage
      for(const auto& statementAccessesPair : statementAccessesPairs_) {
        const auto& access = callerAccesses ? statementAccessesPair->getCallerAccesses()
                                            : statementAccessesPair->getCalleeAccesses();

        // first => AccessID, second => Extent
        for(auto& accessPair : access->getWriteAccesses()) {
          if(!isProvidedByStencilFunctionCall(accessPair.first) &&
             !stencilInstantiation_->isField(accessPair.first))
            continue;

          AccessIDToFieldMap[accessPair.first]->Extent.merge(accessPair.second);
        }

        for(const auto& accessPair : access->getReadAccesses()) {
          if(!isProvidedByStencilFunctionCall(accessPair.first) &&
             !stencilInstantiation_->isField(accessPair.first))
            continue;

          AccessIDToFieldMap[accessPair.first]->Extent.merge(accessPair.second);
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
                               [&](const Field& field) { return field.AccessID == AccessID; });
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
    }
  }

  name += "_" + Interval::makeCodeGenName(stencilFun.getInterval());
  return name;
}

void StencilFunctionInstantiation::setReturn(bool hasReturn) { hasReturn_ = hasReturn; }

bool StencilFunctionInstantiation::hasReturn() const { return hasReturn_; }

bool StencilFunctionInstantiation::isNested() const { return isNested_; }

void StencilFunctionInstantiation::dump() {
  std::cout << "\nStencilFunction : " << getName() << " " << getInterval() << "\n";
  std::cout << MakeIndent<1>::value << "Arguments:\n";

  for(std::size_t argIdx = 0; argIdx < function_->Args.size(); ++argIdx) {

    std::cout << MakeIndent<2>::value << "arg(" << argIdx << ") : ";

    if(isArgOffset(argIdx)) {
      int dim = getCallerOffsetOfArgOffset(argIdx)[0];
      int offset = getCallerOffsetOfArgOffset(argIdx)[1];
      std::cout << "Offset : " << dim2str(dim);
      if(offset != 0)
        std::cout << (offset > 0 ? "+" : "") << offset;
    } else if(isArgField(argIdx)) {
      sir::Field* field = dyn_cast<sir::Field>(function_->Args[argIdx].get());
      std::cout << "Field : " << field->Name << " -> ";
      if(isArgStencilFunctionInstantiation(argIdx)) {
        std::cout << "stencil-function-call:"
                  << getFunctionInstantiationOfArgField(argIdx)->getName();
      } else {
        int callerAccessID = getCallerAccessIDOfArgField(argIdx);
        std::cout << stencilInstantiation_->getNameFromAccessID(callerAccessID) << "  "
                  << getCallerInitialOffsetFromAccessID(callerAccessID);
      }

    } else {
      std::cout << "Direction : " << dim2str(getCallerDimensionOfArgDirection(argIdx));
    }
    std::cout << "\n";
  }

  std::cout << MakeIndent<1>::value << "Accesses (including initial offset):\n";

  const auto& statements = getAST()->getRoot()->getStatements();
  for(std::size_t i = 0; i < statements.size(); ++i) {
    std::cout << "\e[1m" << ASTStringifer::toString(statements[i], 2 * DAWN_PRINT_INDENT) << "\e[0m";
    if(statementAccessesPairs_[i]->getCallerAccesses())
      std::cout << statementAccessesPairs_[i]->getCallerAccesses()->toString(this,
                                                                             3 * DAWN_PRINT_INDENT)
                << "\n";
  }
  std::cout.flush();
}

} // namespace dawn
