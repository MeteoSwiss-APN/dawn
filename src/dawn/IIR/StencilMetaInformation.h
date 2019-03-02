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

#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/StringRef.h"
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
  class VariableVersions {
  public:
    /// Map of AccessIDs to the the list of all AccessIDs of the multi-versioned variables. Note
    /// that the index in the vector corresponds to the version number.
    std::unordered_map<int, std::shared_ptr<std::vector<int>>> variableVersionsMap_;
    std::unordered_map<int, int> versionToOriginalVersionMap_;
    std::unordered_set<int> versionIDs_;

    bool hasVariableMultipleVersions(const int accessID) const {
      return variableVersionsMap_.count(accessID);
    }

    std::shared_ptr<std::vector<int>> getVersions(const int accessID) {
      return variableVersionsMap_.at(accessID);
    }
    const std::shared_ptr<std::vector<int>> getVersions(const int accessID) const {
      return variableVersionsMap_.at(accessID);
    }

    void insert(const int accessID, std::shared_ptr<std::vector<int>> versionsID) {
      variableVersionsMap_.emplace(accessID, versionsID);
      for(auto it : *versionsID) {
        versionIDs_.emplace(it);
        versionToOriginalVersionMap_[it] = accessID;
      }
    }

    bool isAccessIDAVersion(const int accessID) { return versionIDs_.count(accessID); }

    int getOriginalVersionOfAccessID(const int accessID) const {
      return versionToOriginalVersionMap_.at(accessID);
    }
    const std::unordered_set<int>& getVersionIDs() const { return versionIDs_; }

    VariableVersions() = default;

    json::json jsonDump() const;
  };

public:
  /// @brief get the `name` associated with the `accessID` of any access type
  const std::string& getFieldNameFromAccessID(int AccessID) const;

  /// @brief Get the `name` associated with the literal `AccessID`
  const std::string& getNameFromLiteralAccessID(int AccessID) const;

  /// @brief Check whether the `AccessID` corresponds to a literal constant
  inline bool isLiteral(int AccessID) const {
    return AccessID < 0 && LiteralAccessIDToNameMap_.count(AccessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a field
  inline bool isField(int AccessID) const { return FieldAccessIDSet_.count(AccessID); }

  /// @brief check whether the `accessID` is accessed in more than one stencil
  bool isIDAccessedMultipleStencils(int accessID) const;

  /// @brief Check whether the `AccessID` corresponds to a temporary field
  inline bool isTemporaryField(int AccessID) const {
    return isField(AccessID) && TemporaryFieldAccessIDSet_.count(AccessID);
  }

  /// @brief Check whether the `AccessID` corresponds to an accesses of a global variable
  inline bool isGlobalVariable(int AccessID) const {
    return GlobalVariableAccessIDSet_.count(AccessID);
  }
  // TODO who is using this ? Do we need the NameToAccessID because of this?
  bool isGlobalVariable(const std::string& name) const;

  /// @brief Check whether the `AccessID` corresponds to a variable
  bool isVariable(int AccessID) const { return !isField(AccessID) && !isLiteral(AccessID); }

  /// @brief Get the AccessID-to-Name map
  std::unordered_map<std::string, int>& getNameToAccessIDMap();
  const std::unordered_map<std::string, int>& getNameToAccessIDMap() const;

  /// @brief get the `name` associated with the `accessID` of any access type
  std::string getNameFromAccessID(int accessID) const;

  /// @brief Get the `AccessID` associated with the `name`
  ///
  /// Note that this only works for field and variable names, the mapping of literals AccessIDs
  /// and their name is a not bijective!
  int getAccessIDFromName(const std::string& name) const;

  /// @brief Get the field-AccessID set
  const std::set<int>& getFieldAccessIDSet() const;

  /// @brief Get the field-AccessID set
  const std::set<int>& getGlobalVariableAccessIDSet() const;

  /// @brief Get the Literal-AccessID-to-Name map
  const std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap() const;

  /// @brief Get StencilID of the StencilCallDeclStmt
  const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
  getStencilCallToStencilIDMap() const;

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

  /// @brief this checks if the user specialized the field to a dimensionality. If not all
  /// dimensions are allow for off-center acesses and hence, {1,1,1} is returned. If we got a
  /// specialization, it is returned
  Array3i getFieldDimensionsMask(int FieldID) const;

  // TODO rename all these to insert
  /// @brief Set the `AccessID` of the Expr (VarAccess or FieldAccess)
  void setAccessIDOfExpr(const std::shared_ptr<Expr>& expr, const int accessID);

  /// @brief Set the `AccessID` of the Stmt (VarDeclStmt)
  void setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt, const int accessID);

  /// @brief Insert a new AccessID - Name pair
  void setAccessIDNamePair(int accessID, const std::string& name);

  /// @brief Insert a new AccessID - Name pair of a field
  void setAccessIDNamePairOfField(int AccessID, const std::string& name, bool isTemporary = false);

  /// @brief Insert a new AccessID - Name pair of a global variable (i.e scalar field access)
  void setAccessIDNamePairOfGlobalVariable(int AccessID, const std::string& name);

  /// @brief Remove the field, variable or literal given by `AccessID`
  void removeAccessID(int AccesssID);

  /// @brief Add entry to the map between a given expr to its access ID
  void mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID);

  /// @brief Add entry to the map between a given stmt to its access ID
  void mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID);

  void insertLiteralAccessID(const int accessID, const std::string& name);

  /// @brief Add entry of the Expr to AccessID map
  void eraseExprToAccessID(std::shared_ptr<Expr> expr);

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
  /// Map of AccessIDs and to the name of the variable/field. Note that only for fields of the "main
  /// stencil" we can get the AccessID by name. This is due the fact that fields of different
  /// stencil functions can share the same name.
  std::unordered_map<int, std::string> AccessIDToNameMap_;

  /// Can be filled from the AccessIDToName map that is in Metainformation
  std::unordered_map<std::string, int> NameToAccessIDMap_;

  /// Surjection of AST Nodes, Expr (FieldAccessExpr or VarAccessExpr) or Stmt (VarDeclStmt), to
  /// their AccessID. The surjection implies that multiple AST Nodes can have the same AccessID,
  /// which is the intended behaviour as we want to get the same ID back when we access the same
  /// field for example
  std::unordered_map<int, int> ExprIDToAccessIDMap_;
  std::unordered_map<int, int> StmtIDToAccessIDMap_;

  /// Injection of AccessIDs of literal constant to their respective name (usually the name is just
  /// the string representation of the value). Note that literals always have *strictly* negative
  /// AccessIDs, which makes them distinguishable from field or variable AccessIDs. Further keep in
  /// mind that each access to a literal creates a new AccessID!
  std::unordered_map<int, std::string> LiteralAccessIDToNameMap_;

  /// This is a set of AccessIDs which correspond to fields. This allows to fully identify if a
  /// AccessID is a field, variable or literal as literals have always strictly negative IDs and
  /// variables are neither field nor literals.
  std::set<int> FieldAccessIDSet_;

  /// This is an ordered list of IDs of fields that belong to the user API call of the program
  std::vector<int> apiFieldIDs_;

  /// Set containing the AccessIDs of fields which are represented by a temporary storages
  std::set<int> TemporaryFieldAccessIDSet_;

  /// Set containing the AccessIDs of "global variable" accesses. Global variable accesses are
  /// represented by global_accessor or if we know the value at compile time we do a constant
  /// folding of the variable
  std::set<int> GlobalVariableAccessIDSet_;

  /// Map of AccessIDs to the list of all AccessIDs of the multi-versioned field, for fields and
  /// variables
  VariableVersions variableVersions_;

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

  SourceLocation stencilLocation_;

  std::string stencilName_;

  std::string fileName_;

  std::vector<std::shared_ptr<sir::StencilFunction>> allStencilFunctions_;

  StencilMetaInformation() = default;

  json::json jsonDump() const;

  void clone(const StencilMetaInformation& origin);
};
} // namespace iir
} // namespace dawn

#endif
