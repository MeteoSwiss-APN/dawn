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

#ifndef DAWN_IIR_METAINFORMATION_H
#define DAWN_IIR_METAINFORMATION_H

#include "dawn/IIR/Extents.h"
#include "dawn/IIR/FieldAccessMetadata.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/DoubleSidedMap.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/StringRef.h"
#include "dawn/Support/UIDGenerator.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dawn {
namespace iir {
class StencilFunctionInstantiation;

/// @brief Specific instantiation of a stencil
/// @ingroup optimizer
class StencilMetaInformation : public NonCopyable {

public:
  /// @brief get the `name` associated with the `accessID` of any access type
  const std::string& getFieldNameFromAccessID(int AccessID) const;

  /// @brief Get the `name` associated with the literal `AccessID`
  const std::string& getNameFromLiteralAccessID(int AccessID) const;

  bool isAccessType(FieldAccessType fType, const int accessID) const;

  /// @brief Check whether the `AccessID` corresponds to a literal constant
  inline bool isLiteral(int AccessID) const {
    return AccessID < 0 && fieldAccessMetadata_.LiteralAccessIDToNameMap_.count(AccessID);
  }

  /// @brief check whether the `accessID` is accessed in more than one stencil
  bool isIDAccessedMultipleStencils(int accessID) const;

  //  /// @brief Check whether the `AccessID` corresponds to a temporary field
  //  bool isTemporaryField(int AccessID) const {
  //    return isAccessType(FieldAccessType::FAT_MemoryField, AccessID) &&
  //           fieldAccessMetadata_.TemporaryFieldAccessIDSet_.count(AccessID);
  //  }

  /// @brief Check whether the `AccessID` corresponds to an accesses of a global variable
  bool isGlobalVariable(int AccessID) const {
    return fieldAccessMetadata_.GlobalVariableAccessIDSet_.count(AccessID);
  }
  // TODO who is using this ? Do we need the NameToAccessID because of this?
  bool isGlobalVariable(const std::string& name) const;

  /// @brief Check whether the `AccessID` corresponds to a variable
  bool isVariable(int AccessID) const {
    return !isAccessType(FieldAccessType::FAT_MemoryField, AccessID) && !isLiteral(AccessID);
  }

  bool isAccessIDAVersion(const int accessID) {
    return fieldAccessMetadata_.variableVersions_.isAccessIDAVersion(accessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a multi-versioned field
  bool isMultiVersionedField(int AccessID) const {
    return isAccessType(FieldAccessType::FAT_MemoryField, AccessID) &&
           fieldAccessMetadata_.variableVersions_.hasVariableMultipleVersions(AccessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a multi-versioned variable
  bool isMultiVersionedVariable(int AccessID) const {
    return isVariable(AccessID) &&
           fieldAccessMetadata_.variableVersions_.hasVariableMultipleVersions(AccessID);
  }

  int getOriginalVersionOfAccessID(const int accessID) const {
    return fieldAccessMetadata_.variableVersions_.getOriginalVersionOfAccessID(accessID);
  }

  /// @brief Get the AccessID-to-Name map
  const std::unordered_map<std::string, int>& getNameToAccessIDMap() const;

  /// @brief Get the AccessID-to-Name map
  const std::unordered_map<int, std::string>& getAccessIDToNameMap() const;

  /// @brief get the `name` associated with the `accessID` of any access type
  std::string getNameFromAccessID(int accessID) const;

  /// @brief this checks if the user specialized the field to a dimensionality. If not all
  /// dimensions are allow for off-center acesses and hence, {1,1,1} is returned. If we got a
  /// specialization, it is returned
  Array3i getFieldDimensionsMask(int fieldID) const;

  const std::vector<int>& getAPIFieldIDs() const { return fieldAccessMetadata_.apiFieldIDs_; }
  std::vector<int>& getAPIFieldIDs() { return fieldAccessMetadata_.apiFieldIDs_; }

  /// @brief Get the `AccessID` associated with the `name`
  ///
  /// Note that this only works for field and variable names, the mapping of literals AccessIDs
  /// and their name is a not bijective!
  int getAccessIDFromName(const std::string& name) const;

  bool hasNameToAccessID(const std::string& name) const {
    return AccessIDToNameMap_.reverseHas(name);
  }
  /// @brief Get the field-AccessID set
  const std::set<int>& getFieldAccessIDSet() const;

  /// @brief Get the field-AccessID set
  const std::set<int>& getGlobalVariableAccessIDSet() const;

  /// @brief Get the Literal-AccessID-to-Name map
  const std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap() const;

  std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap() {
    return fieldAccessMetadata_.LiteralAccessIDToNameMap_;
  }

  /// @brief Get StencilID of the StencilCallDeclStmt
  const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
  getStencilCallToStencilIDMap() const;
  const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
  getStencilIDToStencilCallMap() const;

  int getStencilIDFromStencilCallStmt(const std::shared_ptr<StencilCallDeclStmt>& stmt) const;

  Extents getBoundaryConditionExtentsFromBCStmt(
      const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) const {
    DAWN_ASSERT_MSG(BoundaryConditionToExtentsMap_.count(stmt),
                    "Boundary Condition does not have a matching Extent");
    return BoundaryConditionToExtentsMap_.at(stmt);
  }

  bool
  hasBoundaryConditionStmtToExtent(const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) const {
    return BoundaryConditionToExtentsMap_.count(stmt);
  }

  void insertBoundaryConditiontoExtentPair(std::shared_ptr<BoundaryConditionDeclStmt>& bc,
                                           Extents& extents) {
    BoundaryConditionToExtentsMap_.emplace(bc, extents);
  }

  /// @brief get a stencil function instantiation by StencilFunCallExpr
  const std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<StencilFunCallExpr>& expr) const;

  /// @brief Get the `AccessID` of the Expr (VarAccess or FieldAccess)
  int getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const;

  /// @brief Get the `AccessID` of the Stmt (VarDeclStmt)
  int getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const;

  const std::vector<std::shared_ptr<StencilFunctionInstantiation>>&
  getStencilFunctionInstantiations() const {
    return stencilFunctionInstantiations_;
  }

  const std::set<int>& getAllocatedFieldAccessIDSet() const {
    return fieldAccessMetadata_.AllocatedFieldAccessIDSet_;
  }
  bool isFieldScope(FieldAccessScope fieldScope, const int accessID) const {
    if(!isAccessType(FieldAccessType::FAT_MemoryField, accessID))
      return false;
    if(fieldScope == FieldAccessScope::FAS_InterStencilTemporary) {
      return fieldAccessMetadata_.AllocatedFieldAccessIDSet_.count(accessID);
    } else if(fieldScope == FieldAccessScope::FAS_StencilTemporary) {
      return fieldAccessMetadata_.TemporaryFieldAccessIDSet_.count(accessID);
    }
  }

  //  bool hasAllocateField(const int accessID) const {
  //    return fieldAccessMetadata_.AllocatedFieldAccessIDSet_.count(accessID);
  //  }
  /// @brief Check if the stencil instantiation needs to allocate fields
  bool hasAllocatedFields() const {
    return !fieldAccessMetadata_.AllocatedFieldAccessIDSet_.empty();
  }

  //  /// @brief Check whether the `AccessID` corresponds to a manually allocated field
  //  bool isAllocatedField(int AccessID) const {
  //    return isAccessType(FieldAccessType::FAT_MemoryField, AccessID) &&
  //    hasAllocateField(AccessID);
  //  }

  void insertAllocatedField(const int accessID);
  void eraseAllocatedField(const int accessID);

  // TODO rename all these to insert
  /// @brief Set the `AccessID` of the Expr (VarAccess or FieldAccess)
  void setAccessIDOfExpr(const std::shared_ptr<Expr>& expr, const int accessID);

  /// @brief Set the `AccessID` of the Stmt (VarDeclStmt)
  void setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt, const int accessID);

  bool hasStmtToAccessID(const std::shared_ptr<Stmt>& stmt) const;

  void insertStmtToAccessID(const std::shared_ptr<Stmt>& stmt, const int accessID);

  /// @brief Insert a new AccessID - Name pair
  void setAccessIDNamePair(int accessID, const std::string& name);

  /// @brief Insert a new AccessID - Name pair of a field
  void setAccessIDNamePairOfField(int AccessID, const std::string& name, bool isTemporary = false);

  /// @brief Insert a new AccessID - Name pair of a global variable (i.e scalar field access)
  void setAccessIDNamePairOfGlobalVariable(int AccessID, const std::string& name);

  void insertStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt, int stencilID);

  /// @brief Remove the field, variable or literal given by `AccessID`
  void removeAccessID(int AccesssID);

  /// @brief Add entry to the map between a given expr to its access ID
  void mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID);

  /// @brief Add entry to the map between a given stmt to its access ID
  void mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID);

  void insertLiteralAccessID(const int accessID, const std::string& name);

  /// @brief Add entry of the Expr to AccessID map
  void eraseExprToAccessID(std::shared_ptr<Expr> expr);

  void eraseStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt);
  void eraseStencilID(const int stencilID);

  ///@brief struct with properties of a stencil function instantiation candidate
  struct StencilFunctionInstantiationCandidate {
    /// stencil function instantiation from where the stencil function instantiation candidate is
    /// called
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunction_;
  };

  // TODO make all these private
  //================================================================================================
  // Stored MetaInformation
  //================================================================================================

  FieldAccessMetadata fieldAccessMetadata_;

  /// Map of AccessIDs and to the name of the variable/field. Note that only for fields of the
  /// "main
  /// stencil" we can get the AccessID by name. This is due the fact that fields of different
  /// stencil functions can share the same name.
  DoubleSidedMap<int, std::string> AccessIDToNameMap_;

  /// Surjection of AST Nodes, Expr (FieldAccessExpr or VarAccessExpr) or Stmt (VarDeclStmt), to
  /// their AccessID. The surjection implies that multiple AST Nodes can have the same AccessID,
  /// which is the intended behaviour as we want to get the same ID back when we access the same
  /// field for example
  std::unordered_map<int, int> ExprIDToAccessIDMap_;
  std::unordered_map<int, int> StmtIDToAccessIDMap_;

  /// Referenced stencil functions in this stencil (note that nested stencil functions are not
  /// stored here but rather in the respecticve `StencilFunctionInstantiation`)
  std::vector<std::shared_ptr<StencilFunctionInstantiation>> stencilFunctionInstantiations_;
  std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                     std::shared_ptr<StencilFunctionInstantiation>>
      ExprToStencilFunctionInstantiationMap_;

  // TODO a set here would be enough
  /// lookup table containing all the stencil function candidates, whose arguments are not yet bound
  std::unordered_map<std::shared_ptr<StencilFunctionInstantiation>,
                     StencilFunctionInstantiationCandidate>
      stencilFunInstantiationCandidate_;

  /// Field Name to BoundaryConditionDeclStmt
  std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>
      FieldnameToBoundaryConditionMap_;

  /// Map of Field ID's to their respecive legal dimensions for offsets if specified in the code
  std::unordered_map<int, Array3i> fieldIDToInitializedDimensionsMap_;

  /// Map of the globally defined variable names to their Values
  std::unordered_map<std::string, std::shared_ptr<sir::Value>> globalVariableMap_;

  /// Can be filled from the StencilIDToStencilCallMap that is in Metainformation
  DoubleSidedMap<int, std::shared_ptr<StencilCallDeclStmt>> StencilIDToStencilCallMap_;

  /// BoundaryConditionCall to Extent Map. Filled my `PassSetBoundaryCondition`
  std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>
      BoundaryConditionToExtentsMap_;

  SourceLocation stencilLocation_;

  std::string stencilName_;

  std::string fileName_;

  StencilMetaInformation() = default;

  json::json jsonDump() const;

  void clone(const StencilMetaInformation& origin);
};
} // namespace iir
} // namespace dawn

#endif
