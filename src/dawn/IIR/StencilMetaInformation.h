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
#include "dawn/IIR/InstantiationHelper.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/DoubleSidedMap.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/RemoveIf.hpp"
#include "dawn/Support/StringRef.h"
#include "dawn/Support/UIDGenerator.h"
#include "dawn/Support/Unreachable.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dawn {
class IIRSerializer;

namespace iir {
class StencilFunctionInstantiation;

/// @brief Specific instantiation of a stencil
/// @ingroup optimizer
class StencilMetaInformation : public NonCopyable {
  friend IIRSerializer;

public:
  StencilMetaInformation(const sir::GlobalVariableMap& globalVariables);

  void clone(const StencilMetaInformation& origin);

  /// @brief get the `name` associated with the `accessID` of any access type
  const std::string& getFieldNameFromAccessID(int AccessID) const;

  /// @brief Get the `name` associated with the literal `AccessID`
  const std::string& getNameFromLiteralAccessID(int AccessID) const;

  bool isAccessType(FieldAccessType fType, const int accessID) const;
  bool isAccessType(FieldAccessType fType, const std::string& name) const;

  /// @brief check whether the `accessID` is accessed in more than one stencil
  bool isIDAccessedMultipleStencils(int accessID) const;

  bool isAccessIDAVersion(const int accessID) {
    return fieldAccessMetadata_.variableVersions_.isAccessIDAVersion(accessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a multi-versioned field
  bool isMultiVersionedField(int AccessID) const {
    return isAccessType(FieldAccessType::FAT_Field, AccessID) &&
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

  template <FieldAccessType TFieldAccessType>
  bool hasAccessesOfType() const {
    return !getAccessesOfType<TFieldAccessType>().empty();
  }

  template <FieldAccessType TFieldAccessType>
  typename TypeOfAccessContainer<TFieldAccessType>::type getAccessesOfType() const {
    return boost::get<const typename TypeOfAccessContainer<TFieldAccessType>::type>(
        getAccessesOfTypeImpl(TFieldAccessType));
  }

  void moveRegisteredFieldTo(FieldAccessType type, int accessID) {
    // we can not move it into an API field, since the original order would not be preserved
    DAWN_ASSERT(type != FieldAccessType::FAT_APIField);
    DAWN_ASSERT_MSG(isFieldType(type), "non field access type can not be moved");

    fieldAccessMetadata_.accessIDType_[accessID] = type;

    if(fieldAccessMetadata_.TemporaryFieldAccessIDSet_.count(accessID)) {
      fieldAccessMetadata_.TemporaryFieldAccessIDSet_.erase(accessID);
    }
    if(fieldAccessMetadata_.AllocatedFieldAccessIDSet_.count(accessID)) {
      fieldAccessMetadata_.AllocatedFieldAccessIDSet_.erase(accessID);
    }

    if(type == FieldAccessType::FAT_StencilTemporary) {
      fieldAccessMetadata_.TemporaryFieldAccessIDSet_.insert(accessID);
    } else if(type == FieldAccessType::FAT_InterStencilTemporary) {
      fieldAccessMetadata_.AllocatedFieldAccessIDSet_.insert(accessID);
    }
  }

  int insertAccessOfType(FieldAccessType type, const std::string& name) {
    int accessID = UIDGenerator::getInstance()->get();
    insertAccessOfType(type, accessID, name);
    return accessID;
  }

  void insertAccessOfType(FieldAccessType type, int AccessID, const std::string& name) {
    setAccessIDNamePair(AccessID, name);
    fieldAccessMetadata_.accessIDType_[AccessID] = type;
    if(isFieldType(type)) {
      fieldAccessMetadata_.FieldAccessIDSet_.insert(AccessID);
      if(type == FieldAccessType::FAT_StencilTemporary) {
        fieldAccessMetadata_.TemporaryFieldAccessIDSet_.insert(AccessID);
      } else if(type == FieldAccessType::FAT_InterStencilTemporary) {
        fieldAccessMetadata_.AllocatedFieldAccessIDSet_.insert(AccessID);
      } else if(type == FieldAccessType::FAT_APIField) {
        fieldAccessMetadata_.apiFieldIDs_.push_back(AccessID);
      }
    } else if(type == FieldAccessType::FAT_GlobalVariable) {
      fieldAccessMetadata_.GlobalVariableAccessIDSet_.insert(AccessID);
    } else if(type == FieldAccessType::FAT_LocalVariable) {
      // local variables are not stored
    } else if(type == FieldAccessType::FAT_Literal) {
      fieldAccessMetadata_.LiteralAccessIDToNameMap_.emplace(AccessID, name);
    }
  }

  void insertField(FieldAccessType type, const std::string& name, const Array3i fieldDimensions);

  int insertStmt(bool keepVarNames, const std::shared_ptr<VarDeclStmt>& stmt) {
    int accessID = UIDGenerator::getInstance()->get();

    std::string globalName;
    if(keepVarNames)
      globalName = stmt->getName();
    else
      globalName = InstantiationHelper::makeLocalVariablename(stmt->getName(), accessID);

    setAccessIDNamePair(accessID, globalName);
    StmtIDToAccessIDMap_.emplace(stmt->getID(), accessID);

    return accessID;
  }

  void eraseStencilFunctionInstantiation(
      const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
    RemoveIf(
        stencilFunctionInstantiations_,
        [&](const std::shared_ptr<StencilFunctionInstantiation>& v) { return (v == stencilFun); });
  }
  void eraseExprToStencilFunction(const std::shared_ptr<StencilFunCallExpr>& expr) {
    ExprToStencilFunctionInstantiationMap_.erase(expr);
  }

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

  /// @brief Set the `AccessID` of the Expr (VarAccess or FieldAccess)
  void setAccessIDOfExpr(const std::shared_ptr<Expr>& expr, const int accessID);

  /// @brief Set the `AccessID` of the Stmt (VarDeclStmt)
  void setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt, const int accessID);

  bool hasStmtToAccessID(const std::shared_ptr<Stmt>& stmt) const;

  void insertStmtToAccessID(const std::shared_ptr<Stmt>& stmt, const int accessID);

  /// @brief Insert a new AccessID - Name pair
  void setAccessIDNamePair(int accessID, const std::string& name);

  void insertStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt, int stencilID);

  /// @brief Remove the field, variable or literal given by `AccessID`
  void removeAccessID(int AccesssID);

  /// @brief Add entry to the map between a given expr to its access ID
  void insertExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID);

  /// @brief Add entry of the Expr to AccessID map
  void eraseExprToAccessID(std::shared_ptr<Expr> expr);

  /// @brief Add entry of the Stmt to AccessID map
  void eraseStmtToAccessID(std::shared_ptr<Stmt> stmt);

  void eraseStencilCallStmt(std::shared_ptr<StencilCallDeclStmt> stmt);
  void eraseStencilID(const int stencilID);

  json::json jsonDump() const;

  ///@brief struct with properties of a stencil function instantiation candidate
  struct StencilFunctionInstantiationCandidate {
    /// stencil function instantiation from where the stencil function instantiation candidate is
    /// called
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunction_;
  };

  std::string getFileName() const { return fileName_; }
  std::string getStencilName() const { return stencilName_; }
  SourceLocation getStencilLocation() const { return stencilLocation_; }

  const std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>&
  getFieldNameToBCMap() const {
    return fieldnameToBoundaryConditionMap_;
  }

  bool hasBC() const { return !fieldnameToBoundaryConditionMap_.empty(); }
  bool hasFieldBC(std::string name) const { return fieldnameToBoundaryConditionMap_.count(name); }
  void insertFieldBC(std::string name, const std::shared_ptr<BoundaryConditionDeclStmt>& bc) {
    fieldnameToBoundaryConditionMap_.emplace(name, bc);
  }

  bool isFieldType(FieldAccessType accessType) const {
    return accessType == FieldAccessType::FAT_Field ||
           accessType == FieldAccessType::FAT_APIField ||
           accessType == FieldAccessType::FAT_StencilTemporary ||
           accessType == FieldAccessType::FAT_InterStencilTemporary;
  }
  void setStencilname(const std::string& name) { stencilName_ = name; }
  void setFileName(const std::string& name) { fileName_ = name; }
  void setStencilLocation(const SourceLocation& location) { stencilLocation_ = location; }

  const FieldAccessMetadata& getFieldAccessMetadata() const { return fieldAccessMetadata_; }

  void insertVersions(const int accessID, std::shared_ptr<std::vector<int>> versionsID);

  bool hasVariableMultipleVersions(const int accessID) const {
    return fieldAccessMetadata_.variableVersions_.hasVariableMultipleVersions(accessID);
  }

  std::shared_ptr<std::vector<int>> getVersionsOf(const int accessID) const;

  const std::unordered_map<int, int>& getExprIDToAccessIDMap() const;
  const std::unordered_map<int, int>& getStmtIDToAccessIDMap() const;

  const std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                           std::shared_ptr<StencilFunctionInstantiation>>&
  getExprToStencilFunctionInstantiation() const {
    return ExprToStencilFunctionInstantiationMap_;
  }

  void insertExprToStencilFunctionInstantiation(
      const std::shared_ptr<StencilFunCallExpr>& expr,
      const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
    ExprToStencilFunctionInstantiationMap_.emplace(expr, stencilFun);
  }

  const std::unordered_map<std::shared_ptr<StencilFunctionInstantiation>,
                           StencilFunctionInstantiationCandidate>&
  getStencilFunInstantiationCandidates() const {
    return stencilFunInstantiationCandidate_;
  }

  void markStencilFunctionInstantiationFinal(
      const std::shared_ptr<StencilFunctionInstantiation>& stencilFun) {
    stencilFunInstantiationCandidate_.erase(stencilFun);
    stencilFunctionInstantiations_.push_back(stencilFun);
  }
  void insertStencilFunctionInstantiation(
      const std::shared_ptr<StencilFunctionInstantiation>& stencilFunctionInstantiation) {
    return stencilFunctionInstantiations_.push_back(stencilFunctionInstantiation);
  }

  void deregisterStencilFunction(std::shared_ptr<StencilFunctionInstantiation> stencilFun) {

    bool found = RemoveIf(ExprToStencilFunctionInstantiationMap_,
                          [&](std::pair<std::shared_ptr<StencilFunCallExpr>,
                                        std::shared_ptr<StencilFunctionInstantiation>>
                                  pair) { return (pair.second == stencilFun); });
    DAWN_ASSERT(found);
    found = RemoveIf(
        stencilFunctionInstantiations_,
        [&](const std::shared_ptr<StencilFunctionInstantiation>& v) { return (v == stencilFun); });
    DAWN_ASSERT(found);
  }

  void insertStencilFunInstantiationCandidate(
      const std::shared_ptr<StencilFunctionInstantiation>& stencilFun,
      const StencilFunctionInstantiationCandidate& candidate) {
    stencilFunInstantiationCandidate_.emplace(stencilFun, candidate);
  }

  const std::unordered_map<int, Array3i>& getFieldIDToDimsMap() const {
    return fieldIDToInitializedDimensionsMap_;
  }

private:
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

  /// lookup table containing all the stencil function candidates, whose arguments are not yet bound
  std::unordered_map<std::shared_ptr<StencilFunctionInstantiation>,
                     StencilFunctionInstantiationCandidate>
      stencilFunInstantiationCandidate_;

  /// Field Name to BoundaryConditionDeclStmt
  std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>
      fieldnameToBoundaryConditionMap_;

  /// Map of Field ID's to their respecive legal dimensions for offsets if specified in the code
  std::unordered_map<int, Array3i> fieldIDToInitializedDimensionsMap_;

  /// Can be filled from the StencilIDToStencilCallMap that is in Metainformation
  DoubleSidedMap<int, std::shared_ptr<StencilCallDeclStmt>> StencilIDToStencilCallMap_;

  /// BoundaryConditionCall to Extent Map. Filled my `PassSetBoundaryCondition`
  std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>
      BoundaryConditionToExtentsMap_;

  SourceLocation stencilLocation_;
  std::string stencilName_;
  std::string fileName_;

  FieldAccessMetadata::allConstContainerTypes
  getAccessesOfTypeImpl(FieldAccessType fieldAccessType) const {
    if(fieldAccessType == FieldAccessType::FAT_Literal) {
      return FieldAccessMetadata::allConstContainerTypes(
          fieldAccessMetadata_.LiteralAccessIDToNameMap_);
    } else if(fieldAccessType == FieldAccessType::FAT_GlobalVariable) {
      return FieldAccessMetadata::allConstContainerTypes(
          fieldAccessMetadata_.GlobalVariableAccessIDSet_);
    } else if(fieldAccessType == FieldAccessType::FAT_Field) {
      return FieldAccessMetadata::allConstContainerTypes(fieldAccessMetadata_.FieldAccessIDSet_);
    } else if(fieldAccessType == FieldAccessType::FAT_LocalVariable) {
      dawn_unreachable("getter of local accesses ids not supported");
    } else if(fieldAccessType == FieldAccessType::FAT_StencilTemporary) {
      return FieldAccessMetadata::allConstContainerTypes(
          fieldAccessMetadata_.TemporaryFieldAccessIDSet_);
    } else if(fieldAccessType == FieldAccessType::FAT_InterStencilTemporary) {
      return FieldAccessMetadata::allConstContainerTypes(
          fieldAccessMetadata_.AllocatedFieldAccessIDSet_);
    } else if(fieldAccessType == FieldAccessType::FAT_APIField) {
      return FieldAccessMetadata::allConstContainerTypes(fieldAccessMetadata_.apiFieldIDs_);
    }
    dawn_unreachable("unknown field access type");
  }
};
} // namespace iir
} // namespace dawn

#endif
