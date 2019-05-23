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

#ifndef DAWN_IIR_STENCILINSTANTIATION_H
#define DAWN_IIR_STENCILINSTANTIATION_H

#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/IIR.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/StringRef.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace dawn {
class OptimizerContext;

namespace iir {

enum class TemporaryScope { TS_LocalVariable, TS_StencilTemporary, TS_Field };

/// @brief Specific instantiation of a stencil
/// @ingroup optimizer
class StencilInstantiation : NonCopyable {

  OptimizerContext* context_;
  StencilMetaInformation metadata_;
  std::unique_ptr<IIR> IIR_;

public:
  /// @brief Assemble StencilInstantiation for stencil
  StencilInstantiation(dawn::OptimizerContext* context);

  StencilMetaInformation& getMetaData();
  const StencilMetaInformation& getMetaData() const { return metadata_; }

  std::shared_ptr<StencilInstantiation> clone() const;

  bool checkTreeConsistency() const;

  /// @brief Get the name of the StencilInstantiation (corresponds to the name of the SIRStencil)
  const std::string getName() const;

  /// @brief Get the orginal `name` and a list of source locations of the field (or variable)
  /// associated with the `AccessID` in the given statement.
  std::pair<std::string, std::vector<SourceLocation>>
  getOriginalNameAndLocationsFromAccessID(int AccessID, const std::shared_ptr<Stmt>& stmt) const;

  /// @brief Get the original name of the field (as registered in the AST)
  std::string getOriginalNameFromAccessID(int AccessID) const;

  /// @brief check whether the `accessID` is accessed in more than one stencil
  bool isIDAccessedMultipleStencils(int accessID) const;

  /// @brief Get the value of the global variable `name`
  const sir::Value& getGlobalVariableValue(const std::string& name) const;

  enum RenameDirection {
    RD_Above, ///< Rename all fields above the current statement
    RD_Below  ///< Rename all fields below the current statement
  };

  /// @brief Add a new version to the field/local variable given by `AccessID`
  ///
  /// This will create a **new** field and trigger a renaming of all the remaining occurences in the
  /// AccessID maps either above or below that statement, starting one statment before or after
  /// the current statement. Optionally, an `Expr` can be passed which will be renamed as well
  /// (usually the left- or right-hand side of an assignment).
  /// Consider the following example:
  ///
  /// @code
  ///   v = 2 * u
  ///   lap = u(i+1)
  ///   u = lap(i+1)
  /// @endcode
  ///
  /// We may want to rename `u` in the second statement (an all occurences of `u` above) to
  /// resolve the race-condition. We expect to end up with:
  ///
  /// @code
  ///   v = 2 * u_1
  ///   lap = u_1(i+1)
  ///   u = lap(i+1)
  /// @endcode
  ///
  /// where `u_1` is the newly created version of `u`.
  ///
  /// @param AccessID   AccessID of the field for which a new version will be created
  /// @param stencil    Current stencil
  /// @param stageIdx   **Linear** index of the stage in the stencil
  /// @param stmtIdx    Index of the statement inside the stage
  /// @param expr       Expression to be renamed (usually the left- or right-hand side of an
  ///                   assignment). Can be `NULL`.
  /// @returns AccessID of the new field
  int createVersionAndRename(int AccessID, Stencil* stencil, int stageIndex, int stmtIndex,
                             std::shared_ptr<Expr>& expr, RenameDirection dir);

  /// @brief Rename all occurences of field `oldAccessID` to `newAccessID`
  void renameAllOccurrences(Stencil* stencil, int oldAccessID, int newAccessID);

  /// @brief Promote the local variable, given by `AccessID`, to a temporary field
  ///
  /// This will take care of registering the new field (and removing the variable) as well as
  /// replacing the variable accesses with point-wise field accesses.
  void promoteLocalVariableToTemporaryField(Stencil* stencil, int AccessID,
                                            const Stencil::Lifetime& lifetime,
                                            TemporaryScope temporaryScope);

  /// @brief Promote the temporary field, given by `AccessID`, to a real storage which needs to be
  /// allocated by the stencil
  void promoteTemporaryFieldToAllocatedField(int AccessID);

  /// @brief Demote the temporary field, given by `AccessID`, to a local variable
  ///
  /// This will take care of registering the new variable (and removing the field) as well as
  /// replacing the field accesses with varible accesses.
  ///
  /// This implicitcly assumes the first access (i.e `lifetime.Begin`) to the field is an
  /// `ExprStmt` and the field is accessed as the LHS of an `AssignmentExpr`.
  void demoteTemporaryFieldToLocalVariable(Stencil* stencil, int AccessID,
                                           const Stencil::Lifetime& lifetime);

  /// @brief Get the list of stencils
  inline const std::vector<std::unique_ptr<Stencil>>& getStencils() const {
    return getIIR()->getChildren();
  }

  /// @brief get the IIR tree
  inline const std::unique_ptr<IIR>& getIIR() const { return IIR_; }

  /// @brief get the IIR tree
  inline std::unique_ptr<IIR>& getIIR() { return IIR_; }

  /// @brief Get the optimizer context
  inline ::dawn::OptimizerContext* getOptimizerContext() { return context_; }

  /// @brief Get the optimizer context
  const ::dawn::OptimizerContext* getOptimizerContext() const { return context_; }

  bool insertBoundaryConditions(std::string originalFieldName,
                                std::shared_ptr<BoundaryConditionDeclStmt> bc);

  /// @brief Get a unique (positive) identifier
  inline int nextUID() { return UIDGenerator::getInstance()->get(); }

  /// @brief Dump the StencilInstantiation to stdout
  void jsonDump(std::string filename) const;

  /// @brief Register a new stencil function
  ///
  /// If `curStencilFunctionInstantiation` is not NULL, the stencil function is treated as a nested
  /// stencil function.
  std::shared_ptr<StencilFunctionInstantiation> makeStencilFunctionInstantiation(
      const std::shared_ptr<StencilFunCallExpr>& expr,
      const std::shared_ptr<sir::StencilFunction>& SIRStencilFun, const std::shared_ptr<AST>& ast,
      const Interval& interval,
      const std::shared_ptr<StencilFunctionInstantiation>& curStencilFunctionInstantiation);

  /// @brief Report the accesses to the console (according to `-freport-accesses`)
  void reportAccesses() const;
};
} // namespace iir
} // namespace dawn

#endif
