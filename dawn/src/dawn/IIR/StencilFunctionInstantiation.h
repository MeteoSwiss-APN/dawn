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

#pragma once

#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/Field.h"
#include "dawn/IIR/Interval.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Unreachable.h"
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace dawn {

namespace iir {

class StencilInstantiation;

inline std::string dim2str(int dim) {
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
}

/// @brief Specific instantiation of a stencil function
///
/// Fields of a stencil function instantiation are mappend to the fields of the @b caller i.e all
/// fields are mapped to one of the fields of the "main" stencil. Arguments are identified by their
/// index in the argument list. For each argument of the function we record the following
/// quantities, depending on their type:
///
///   - Argument is a direction: We record a mapping of the arugment index to the dimension the
///     direction maps to @see getCallerDimensionOfArgDirection().
///   - Argument is an offset: We record a mapping of the argument index to the dimension and
///     dimensional offset @see getCallerOffsetOfArgOffset().
///   - Argument is a field: We record a maping of the argument index to the AccessID the field
///     maps to @see getCallerAccessIDOfArgField().
///
/// In addition, we store the initial offset of each field as specified in the argument list (e.g
/// calling `avg(u(i+1))` will yield an initial offset [1, 0, 0] for @b each access to the field `u`
/// maps to).
///
/// @b Example:
/// Consider the following example (given in the gtclang DSL):
///
/// @code
/// stencil_function avg {
///   storage in;
///   direction dir;
///
///   Do { return in(dir+1); }
/// };
/// @endcode
///
/// Now, we call `avg` with field `u` (which we assume `u` has AccessID `5`):
///
/// @code
/// out = avg(u(i+1), i);
/// @endcode
///
/// The members of the `StencilFunctionInstantiation` for this call:
///
///  Member                              | Values         | Explanation
///  :-----------------------------------| :--------------| :-----------------------------------
///  ArgumentIndexToCallerAccessIDMap    | {0, 5}         | Field `in` maps to field `u`
///  ArgumentIndexToCallerDirectionMap   | {1, 0}         | Direction `dir` maps to dimension 0
///  ArgumentIndexToCallerOffsetMap      | {}             | No offsets
///  CallerAcceessIDToInitialOffsetMap   | {5, [1, 0, 0]} | `avg` is called with `u` at `i+1`
///
/// @ingroup optimizer
class StencilFunctionInstantiation {
private:
  StencilInstantiation* stencilInstantiation_;
  const StencilMetaInformation& metadata_;
  // TODO put all members in a struct to avoid having to implement a clone for all of them
  // except the vector<unique_ptr>
  std::shared_ptr<iir::StencilFunCallExpr> expr_;
  std::shared_ptr<sir::StencilFunction> function_;
  std::shared_ptr<iir::AST> ast_;

  Interval interval_;
  bool hasReturn_;
  bool isNested_;

  bool argsBound_ = false;
  //===----------------------------------------------------------------------------------------===//
  //     Argument Maps

  /// Map of the argument index to the field AccessID of the *caller*
  std::unordered_map<int, int> ArgumentIndexToCallerAccessIDMap_;

  /// Map of the argument index to the stencil function (which provides a field) (e.g in
  /// `foo(u, bar(...)` we store the instantiation of bar
  std::unordered_map<int, std::shared_ptr<StencilFunctionInstantiation>>
      ArgumentIndexToStencilFunctionInstantiationMap_;

  /// Map of the argument index to resolved direction (i.e the dimension) of the *caller*
  std::unordered_map<int, int> ArgumentIndexToCallerDirectionMap_;

  /// Map of the argument index to resolved offset (i.e the dimension plus offset) of the *caller*
  std::unordered_map<int, Array2i> ArgumentIndexToCallerOffsetMap_;

  /// Map of *caller* AccessID to the initial offset of the field (e.g the initial offset of the
  /// field mapping to the first argument in `avg(in(i+1))` would be [1, 0, 0])
  std::unordered_map<int, ast::Offsets> CallerAccessIDToInitialOffsetMap_;

  /// Caller AccessID to name
  std::unordered_map<int, std::string> AccessIDToNameMap_;
  std::unordered_map<int, std::string> LiteralAccessIDToNameMap_;

  /// Referenced stencil functions within this stencil function
  std::unordered_map<std::shared_ptr<iir::StencilFunCallExpr>,
                     std::shared_ptr<StencilFunctionInstantiation>>
      ExprToStencilFunctionInstantiationMap_;

  //===----------------------------------------------------------------------------------------===//
  //     Accesses & Fields

  /// DoMethod containing the list of statements in this stencil function
  std::unique_ptr<DoMethod> doMethod_;

  std::vector<Field> calleeFields_;
  std::vector<Field> callerFields_;

  /// Set of AccessID of fields which are not used
  std::set<int> unusedFields_;

  /// Set containing the AccessIDs of "global variable" accesses. Global variable accesses are
  /// represented by global_accessor or if we know the value at compile time we do a constant
  /// folding of the variable
  std::set<int> GlobalVariableAccessIDSet_;

public:
  StencilFunctionInstantiation(StencilInstantiation* context,
                               const std::shared_ptr<iir::StencilFunCallExpr>& expr,
                               const std::shared_ptr<sir::StencilFunction>& function,
                               const std::shared_ptr<iir::AST>& ast, const Interval& interval,
                               bool isNested);

  StencilFunctionInstantiation(StencilFunctionInstantiation&&) = default;

  StencilFunctionInstantiation clone() const;

  inline const std::unique_ptr<DoMethod>& getDoMethod() { return doMethod_; }

  std::unordered_map<int, int>& ArgumentIndexToCallerAccessIDMap() {
    return ArgumentIndexToCallerAccessIDMap_;
  }
  std::unordered_map<int, int> const& ArgumentIndexToCallerAccessIDMap() const {
    return ArgumentIndexToCallerAccessIDMap_;
  }

  /// @brief check if all the stencil function arguments are bound
  bool isArgsBound() const { return argsBound_; }

  /// @brief returns number of arguments
  size_t numArgs() const;

  /// @brief register the access id of a global variable access
  void setAccessIDOfGlobalVariable(int AccessID);

  /// @brief get the access id set of a global variables
  /// @{
  std::set<int>& getAccessIDSetGlobalVariables() { return GlobalVariableAccessIDSet_; }
  std::set<int> const& getAccessIDSetGlobalVariables() const { return GlobalVariableAccessIDSet_; }
  /// @}

  bool hasGlobalVariables() const { return !GlobalVariableAccessIDSet_.empty(); }

  /// @brief remove a stencil function instantiation tagged by a StencilFunCallExpr
  void removeStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr);

  /// @brief get the name of the arg parameter of the stencil function which is called passing
  /// another function
  ///
  /// In the following example
  /// stencil_function fn {
  ///   storage arg_st1, arg_st2;
  /// }
  /// fn(storage1, fn(storage2) )
  /// it will return arg_st2
  std::string getArgNameFromFunctionCall(std::string fnCallName) const;

  /// @brief Get the associated StencilInstantiation
  StencilInstantiation* getStencilInstantiation() { return stencilInstantiation_; }
  const StencilInstantiation* getStencilInstantiation() const { return stencilInstantiation_; }

  /// @brief Get the SIR stencil function
  std::shared_ptr<sir::StencilFunction> getStencilFunction() const { return function_; }

  void setStencilFunction(std::shared_ptr<sir::StencilFunction> fun) { function_ = fun; }

  /// @brief Get the name of the stencil function
  const std::string& getName() const { return function_->Name; }

  /// @brief Get the AST of the stencil function
  std::shared_ptr<iir::AST>& getAST() { return ast_; }
  const std::shared_ptr<iir::AST>& getAST() const { return ast_; }

  /// @brief Evaluate the offset of the field access expression (this performs the lazy evaluation
  /// of the offsets)
  ast::Offsets evalOffsetOfFieldAccessExpr(const std::shared_ptr<iir::FieldAccessExpr>& expr,
                                           bool applyInitialOffset = true) const;

  /// @brief returns true if the argument in the argumentIndex position is bound to an offset
  /// argument
  bool isArgBoundAsOffset(int argumentIndex) const;

  /// @brief returns true if the argument in the argumentIndex position is bound to a direction
  /// argument
  bool isArgBoundAsDirection(int argumentIndex) const;

  /// @brief returns true if the argument in the argumentIndex position is bound to a stencil
  /// function argument
  bool isArgBoundAsFunctionInstantiation(int argumentIndex) const;

  /// @brief returns true if the argument in the argumentIndex position is bound to a field argument
  bool isArgBoundAsFieldAccess(int argumentIndex) const;

  //===----------------------------------------------------------------------------------------===//
  //     Argument Maps
  //===----------------------------------------------------------------------------------------===//

  /// @brief Get/Set the instantiated dimension of the caller of the direction argument given the
  /// argument index
  /// @{
  int getCallerDimensionOfArgDirection(int argumentIndex) const;
  void setCallerDimensionOfArgDirection(int argumentIndex, int dimension);
  /// @}

  /// @brief Get/Set the instantiated dimension/offset pair of the caller of the offset argument
  /// given the argument index
  /// @{
  const Array2i& getCallerOffsetOfArgOffset(int argumentIndex) const;
  void setCallerOffsetOfArgOffset(int argumentIndex, const Array2i& offset);
  /// @}

  /// @brief Get/Set the field AccessID of the @b caller corresponding to the argument index
  /// @{
  int getCallerAccessIDOfArgField(int argumentIndex) const;
  void setCallerAccessIDOfArgField(int argumentIndex, int callerAccessID);
  /// @}

  /// @brief Get/Set the StencilFunctionInstantiation (which provides the field via return) of the
  /// field corresponding to the argument index
  /// @{
  std::shared_ptr<StencilFunctionInstantiation>
  getFunctionInstantiationOfArgField(int argumentIndex) const;
  void
  setFunctionInstantiationOfArgField(int argumentIndex,
                                     const std::shared_ptr<StencilFunctionInstantiation>& func);
  /// @}

  /// @brief Get/Set the initial offset of the @b caller given the caller AccessID
  /// @{
  const ast::Offsets& getCallerInitialOffsetFromAccessID(int callerAccessID) const;
  void setCallerInitialOffsetFromAccessID(int callerAccessID, const ast::Offsets& offset);
  /// @}

  /// @brief Get/Set if a field (given by its AccessID) is provided via a stencil function call
  /// @{
  bool isProvidedByStencilFunctionCall(int callerAccessID) const;
  /// @}

  /// @brief Get the argument index of the field (or stencil function instantiation) given the
  /// AccessID
  int getArgumentIndexFromCallerAccessID(int callerAccessID) const;

  /// @brief Get the original `name` associated with the caller `AccessID`
  const std::string& getOriginalNameFromCallerAccessID(int callerAccessID) const;

  /// @brief Get the arguments of the stencil function
  std::vector<std::shared_ptr<sir::StencilFunctionArg>>& getArguments();
  const std::vector<std::shared_ptr<sir::StencilFunctionArg>>& getArguments() const;

  /// @brief Check if the argument at the given index is an offset
  bool isArgOffset(int argumentIndex) const;

  /// @brief Check if the argument at the given index is a direction
  bool isArgDirection(int argumentIndex) const;

  /// @brief Check if the argument at the given index is a field
  bool isArgField(int argumentIndex) const;

  /// @brief Check if the argument at the given index is a stencil function
  bool isArgStencilFunctionInstantiation(int argumentIndex) const;

  /// @brief determines if accessid corresponds to a literal
  bool isLiteral(int accessID) const { return accessID < 0; }

  //===----------------------------------------------------------------------------------------===//
  //     Expr/Stmt to Caller AccessID Maps
  //===----------------------------------------------------------------------------------------===//

  /// @brief Get the `name` associated with the `AccessID` of a Field or a Var
  std::string getFieldNameFromAccessID(int AccessID) const;

  /// @brief Get the `name` associated with the `AccessID` of any access type
  std::string getNameFromAccessID(int accessID) const;

  /// @brief Get the `name` associated with the literal `AccessID`
  const std::string& getNameFromLiteralAccessID(int AccessID) const;

  /// @brief Get the Literal-AccessID-to-Name map
  std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap();
  const std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap() const;

  /// @brief Get the AccessID-to-Name map
  std::unordered_map<int, std::string>& getAccessIDToNameMap();
  const std::unordered_map<int, std::string>& getAccessIDToNameMap() const;

  std::unordered_map<int, ast::Offsets>& getCallerAccessIDToInitialOffsetMap();
  const std::unordered_map<int, ast::Offsets>& getCallerAccessIDToInitialOffsetMap() const;

  /// @brief Get StencilFunctionInstantiation of the `StencilFunCallExpr`
  const std::unordered_map<std::shared_ptr<iir::StencilFunCallExpr>,
                           std::shared_ptr<StencilFunctionInstantiation>>&
  getExprToStencilFunctionInstantiationMap() const;

  /// @brief Get StencilFunctionInstantiation of the `StencilFunCallExpr`
  std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr) const;

  bool hasStencilFunctionInstantiation(const std::shared_ptr<iir::StencilFunCallExpr>& expr) const;

  void insertExprToStencilFunction(const std::shared_ptr<StencilFunctionInstantiation>& stencilFun);

  template <class MapType, class KeyType>
  static void replaceKeyInMap(MapType& map, KeyType oldKey, KeyType newKey) {
    auto it = map.find(oldKey);
    if(it != map.end()) {
      map.emplace(newKey, std::move(it->second));
      map.erase(it);
    }
  }

  //===----------------------------------------------------------------------------------------===//
  //     Accesses & Fields
  //===----------------------------------------------------------------------------------------===//

  /// @brief Get the statements of the stencil function
  const std::vector<std::shared_ptr<iir::Stmt>>& getStatements() const;

  /// @brief Update the fields and global variables
  ///
  /// This recomputes the fields referenced in this stencil function and computes the @b
  /// accumulated extent of each field
  void update();

  /// @brief Get the field declaration of the field corresponding to the argument index
  const Field& getCallerFieldFromArgumentIndex(int argumentIndex) const;

  /// @brief Get `caller` field declarations needed for this Stage
  const std::vector<Field>& getCallerFields() const;

  /// @brief Get `callee` field declarations needed for this Stage
  const std::vector<Field>& getCalleeFields() const;

  /// @brief Check if field with `AccessID` is unused
  bool isFieldUnused(int AccessID) const;

  //===----------------------------------------------------------------------------------------===//
  //     Miscellaneous
  //===----------------------------------------------------------------------------------------===//

  /// @brief Generate a name uniquely identifying the instantiation of the stencil function for
  /// code generation
  static std::string makeCodeGenName(const StencilFunctionInstantiation& stencilFun);

  /// @brief Get the vertical Interval
  Interval& getInterval() { return interval_; }
  const Interval& getInterval() const { return interval_; }

  /// @brief Set if the stencil function has a return statement
  void setReturn(bool hasReturn);

  /// @brief Check if the stencil function has a return statement
  bool hasReturn() const;

  /// @brief Is this a nested stencil function (i.e called in the argument list of another stencil
  /// function)?
  bool isNested() const;

  /// @brief Get the underlying AST stencil function call expression
  const std::shared_ptr<iir::StencilFunCallExpr>& getExpression() const { return expr_; }

  /// @brief Set the underlying AST stencil function call expression
  void setExpression(const std::shared_ptr<iir::StencilFunCallExpr>& expr) { expr_ = expr; }

  /// @brief finalizes the binding of the arguments of a stencil function.
  /// In particular it associates new accessIDs of arguments that are nested stencil function calls
  void closeFunctionBindings(const std::vector<int>&);

  void closeFunctionBindings();

  /// @brief that all the function bindings are properly set
  void checkFunctionBindings() const;

  /// @brief Dump the stencil function instantiation to stdout
  void dump(std::ostream& os) const;
};

} // namespace iir
} // namespace dawn
