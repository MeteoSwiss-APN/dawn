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

#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/NonCopyable.h"
#include "dawn/Support/StringRef.h"
#include "dawn/Support/UIDGenerator.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace dawn {
namespace iir {

/// @brief Specific instantiation of a stencil
/// @ingroup optimizer
class StencilMetaInformation : NonCopyable {

  ///@brief struct with properties of a stencil function instantiation candidate
  struct StencilFunctionInstantiationCandidate {
    /// stencil function instantiation from where the stencil function instantiation candidate is
    /// called
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunction_;
  };

  ///
  /// @brief The VariableVersions class
  /// It holds all the relevant information of versioned fields: Which are the versioned fields and
  /// maps that show relationships between verioned variables and their original counterparts
  ///
  class VariableVersions {
  private:
    /// Map of AccessIDs to the the list of all AccessIDs of the multi-versioned variables. Note
    /// that the index in the vector corresponds to the version number:
    /// Original Field -> [All the Versioned Fields]
    std::unordered_map<int, std::shared_ptr<std::vector<int>>> variableVersionsMap_;
    /// The map that represents the relationship Versioned field -> Original Field
    std::unordered_map<int, int> versionToOriginalVersionMap_;
    /// A set of all the variables where verisoning was applied
    std::unordered_set<int> versionIDs_;

  public:
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
    VariableVersions() = default;
  };

  /// Unique identifier generator
  UIDGenerator UIDGen_;

  /// Map of AccessIDs and to the name of the variable/field. Note that only for fields of the "main
  /// stencil" we can get the AccessID by name. This is due the fact that fields of different
  /// stencil functions can share the same name.
  std::unordered_map<std::string, int> NameToAccessIDMap_;
  std::unordered_map<int, std::string> AccessIDToNameMap_;

  /// Surjection of AST Nodes, Expr (FieldAccessExpr or VarAccessExpr) or Stmt (VarDeclStmt), to
  /// their AccessID. The surjection implies that multiple AST Nodes can have the same AccessID,
  /// which is the intended behaviour as we want to get the same ID back when we access the same
  /// field for example
  std::unordered_map<std::shared_ptr<Expr>, int> ExprToAccessIDMap_;
  std::unordered_map<std::shared_ptr<Stmt>, int> StmtToAccessIDMap_;

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

  /// Set containing the AccessIDs of fields which are manually allocated by the stencil and serve
  /// as temporaries spanning over multiple stencils
  std::set<int> AllocatedFieldAccessIDSet_;

  /// Set containing the AccessIDs of "global variable" accesses. Global variable accesses are
  /// represented by global_accessor or if we know the value at compile time we do a constant
  /// folding of the variable
  std::set<int> GlobalVariableAccessIDSet_;

  /// Map of AccessIDs to the list of all AccessIDs of the multi-versioned field, for fields and
  /// variables
  VariableVersions variableVersions_;

  /// Stencil description statements. These are built from the StencilDescAst of the sir::Stencil
  std::vector<std::shared_ptr<Statement>> stencilDescStatements_;
  std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int> StencilCallToStencilIDMap_;
  std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>> IDToStencilCallMap_;

  /// StageID to name Map. Filled by the `PassSetStageName`.
  std::unordered_map<int, std::string> StageIDToNameMap_;

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

  /// BoundaryConditionCall to Extent Map. Filled my `PassSetBoundaryCondition`
  std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>
      BoundaryConditionToExtentsMap_;

  /// Field Name to BoundaryConditionDeclStmt
  std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>
      FieldnameToBoundaryConditionMap_;

  /// Set of all the IDs that are locally cached
  std::set<int> CachedVariableSet_;

  /// Map of Field ID's to their respecive legal dimensions for offsets if specified in the code
  std::unordered_map<int, Array3i> fieldIDToInitializedDimensionsMap_;

  /// Map of the globally defined variable names to their Values
  std::unordered_map<std::string, std::shared_ptr<sir::Value>> globalVariableMap_;

  SourceLocation stencilLocation_;

  std::string stencilName_;

  std::string fileName_;

public:
  StencilMetaInformation();

  /// @brief Insert a new AccessID - Name pair
  void setAccessIDNamePair(int AccessID, const std::string& name);

  /// @brief Insert a new AccessID - Name pair of a field
  void setAccessIDNamePairOfField(int AccessID, const std::string& name, bool isTemporary = false);

  /// @brief Insert a new AccessID - Name pair of a global variable (i.e scalar field access)
  void setAccessIDNamePairOfGlobalVariable(int AccessID, const std::string& name);

  /// @brief Remove the field, variable or literal given by `AccessID`
  void removeAccessID(int AccesssID);

  /// @brief Get the `name` associated with the literal `AccessID`
  const std::string& getNameFromLiteralAccessID(int AccessID) const;

  /// @brief Check whether the `AccessID` corresponds to a field
  inline bool isField(int AccessID) const { return FieldAccessIDSet_.count(AccessID); }

  /// @brief Check whether the `AccessID` corresponds to a temporary field
  inline bool isTemporaryField(int AccessID) const {
    return isField(AccessID) && TemporaryFieldAccessIDSet_.count(AccessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a manually allocated field
  inline bool isAllocatedField(int AccessID) const {
    return isField(AccessID) && AllocatedFieldAccessIDSet_.count(AccessID);
  }

  /// @brief Get the set of fields which need to be allocated
  inline const std::set<int>& getAllocatedFieldAccessIDs() const {
    return AllocatedFieldAccessIDSet_;
  }

  /// @brief Check if the stencil instantiation needs to allocate fields
  inline bool hasAllocatedFields() const { return !AllocatedFieldAccessIDSet_.empty(); }

  /// @brief Check whether the `AccessID` corresponds to an accesses of a global variable
  inline bool isGlobalVariable(int AccessID) const {
    return GlobalVariableAccessIDSet_.count(AccessID);
  }
  bool isGlobalVariable(const std::string& name) const;

  /// @brief Get the value of the global variable `name`
  const sir::Value& getGlobalVariableValue(const std::string& name) const;

  /// @brief Check whether the `AccessID` corresponds to a variable
  inline bool isVariable(int AccessID) const { return !isField(AccessID) && !isLiteral(AccessID); }

  /// @brief Check whether the `AccessID` corresponds to a literal constant
  inline bool isLiteral(int AccessID) const {
    return AccessID < 0 && LiteralAccessIDToNameMap_.count(AccessID);
  }

  inline bool isAccessIDAVersion(const int accessID) {
    return variableVersions_.isAccessIDAVersion(accessID);
  }

  inline int getOriginalVersionOfAccessID(const int accessID) const {
    return variableVersions_.getOriginalVersionOfAccessID(accessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a multi-versioned field
  inline bool isMultiVersionedField(int AccessID) const {
    return isField(AccessID) && variableVersions_.hasVariableMultipleVersions(AccessID);
  }

  /// @brief Check whether the `AccessID` corresponds to a multi-versioned variable
  inline bool isMultiVersionedVariable(int AccessID) const {
    return isVariable(AccessID) && variableVersions_.hasVariableMultipleVersions(AccessID);
  }

  /// @brief Get a list of all field AccessIDs of this multi-versioned field
  ArrayRef<int> getFieldVersions(int AccessID) const;

  /// @brief Get the `AccessID` associated with the `name`
  ///
  /// Note that this only works for field and variable names, the mapping of literals AccessIDs
  /// and their name is a not bijective!
  int getAccessIDFromName(const std::string& name) const;

  /// @brief Get the `AccessID` of the Expr (VarAccess or FieldAccess)
  int getAccessIDFromExpr(const std::shared_ptr<Expr>& expr) const;

  /// @brief Set the `AccessID` of the Expr (VarAccess or FieldAccess)
  void setAccessIDOfExpr(const std::shared_ptr<Expr>& expr, const int accessID);

  /// @brief Set the `AccessID` of the Stmt (VarDeclStmt)
  void setAccessIDOfStmt(const std::shared_ptr<Stmt>& stmt, const int accessID);

  /// @brief Get the `AccessID` of the Stmt (VarDeclStmt)
  int getAccessIDFromStmt(const std::shared_ptr<Stmt>& stmt) const;

  /// @brief get a stencil function instantiation by StencilFunCallExpr
  const std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiation(const std::shared_ptr<StencilFunCallExpr>& expr) const;

  /// @brief get a stencil function candidate by StencilFunCallExpr
  std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiationCandidate(const std::shared_ptr<StencilFunCallExpr>& expr);

  /// @brief get a stencil function candidate by name
  std::shared_ptr<StencilFunctionInstantiation>
  getStencilFunctionInstantiationCandidate(const std::string stencilFunName);

  /// @brief Get the list of access ID of the user API fields
  inline std::vector<int>& getAPIFieldIDs() { return apiFieldIDs_; }

  /// @brief clone a stencil function candidate and set its name fo functionName
  /// @returns the clone of the stencil function
  std::shared_ptr<StencilFunctionInstantiation>
  cloneStencilFunctionCandidate(const std::shared_ptr<StencilFunctionInstantiation>& stencilFun,
                                std::string functionName);

  /// @brief Add entry to the map between a given expr to its access ID
  void mapExprToAccessID(const std::shared_ptr<Expr>& expr, int accessID);

  /// @brief Add entry to the map between a given stmt to its access ID
  void mapStmtToAccessID(const std::shared_ptr<Stmt>& stmt, int accessID);

  /// @brief Add entry of the Expr to AccessID map
  void eraseExprToAccessID(std::shared_ptr<Expr> expr);

  /// @brief Get the StencilID of the StencilCallDeclStmt `stmt`
  int getStencilIDFromStmt(const std::shared_ptr<StencilCallDeclStmt>& stmt) const;

  bool insertBoundaryConditions(std::string originalFieldName,
                                std::shared_ptr<BoundaryConditionDeclStmt> bc);

  /// @brief Get a unique (positive) identifier
  inline int nextUID() { return UIDGen_.get(); }

  /// @brief Generate a unique name for a local variable
  static std::string makeLocalVariablename(const std::string& name, int AccessID);

  /// @brief Generate a unique name for a temporary field
  static std::string makeTemporaryFieldname(const std::string& name, int AccessID);

  /// @brief Extract the name of a local variable
  ///
  /// Reverse the effect of `makeLocalVariablename`.
  static std::string extractLocalVariablename(const std::string& name);

  /// @brief Extract the name of a local variable
  ///
  /// Reverse the effect of `makeTemporaryFieldname`.
  static std::string extractTemporaryFieldname(const std::string& name);

  /// @brief Name used for all `StencilCallDeclStmt` in the stencil description AST
  /// (`getStencilDescStatements`) to signal code-gen it should insert a call to the gridtools
  /// stencil here
  static std::string makeStencilCallCodeGenName(int StencilID);

  /// @brief Check if the given name of a `StencilCallDeclStmt` was generate by
  /// `makeStencilCallCodeGenName`
  static bool isStencilCallCodeGenName(const std::string& name);

  /// @brief this checks if the user specialized the field to a dimensionality. If not all
  /// dimensions are allow for off-center acesses and hence, {1,1,1} is returned. If we got a
  /// specialization, it is returned
  Array3i getFieldDimensionsMask(int FieldID);

  inline void insertBoundaryConditiontoExtentPair(std::shared_ptr<BoundaryConditionDeclStmt>& bc,
                                                  Extents& extents) {
    BoundaryConditionToExtentsMap_.emplace(bc, extents);
  }

  inline Extents getBoundaryConditionExtentsFromBCStmt(
      const std::shared_ptr<BoundaryConditionDeclStmt>& stmt) const {
    if(BoundaryConditionToExtentsMap_.count(stmt) == 0) {
      DAWN_ASSERT_MSG(false, "Boundary Condition does not have a matching Extent");
    }
    return BoundaryConditionToExtentsMap_.find(stmt)->second;
  }

  //====--------------------------------------------------------------------------------------------
  // Basic getters and setters
  //===---------------------------------------------------------------------------------------------
  const std::set<int>& getCachedVariableSet() const;

  void insertCachedVariable(int fieldID);

  inline std::unordered_map<int, Array3i>& getFieldIDToInitializedDimensionsMap() {
    return fieldIDToInitializedDimensionsMap_;
  }

  /// @brief Get the name of the StencilInstantiation (corresponds to the name of the SIRStencil)
  const std::string getName() const { return stencilName_; }

  /// @brief Get the `name` associated with the `AccessID`
  const std::string& getNameFromAccessID(int AccessID) const;

  /// @brief Get the `name` associated with the `StageID`
  const std::string& getNameFromStageID(int StageID) const;

  /// @brief Get the orginal `name` and a list of source locations of the field (or variable)
  /// associated with the `AccessID` in the given statement.
  std::pair<std::string, std::vector<SourceLocation>>
  getOriginalNameAndLocationsFromAccessID(int AccessID, const std::shared_ptr<Stmt>& stmt) const;

  inline const std::unordered_map<std::string, std::shared_ptr<sir::Value>>&
  getGlobalVariableMap() const;

  /// @brief Get StencilFunctionInstantiation of the `StencilFunCallExpr`
  const std::unordered_map<std::shared_ptr<StencilFunCallExpr>,
                           std::shared_ptr<StencilFunctionInstantiation>>&
  getExprToStencilFunctionInstantiationMap() const;

  /// @brief Get StencilID of the StencilCallDeclStmt
  std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>& getStencilCallToStencilIDMap();
  const std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>&
  getStencilCallToStencilIDMap() const;

  /// @brief Get StencilID of the StencilCallDeclStmt
  std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>& getIDToStencilCallMap();
  const std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>>&
  getIDToStencilCallMap() const;

  /// @brief Get the stencil description AST
  inline const std::vector<std::shared_ptr<Statement>>& getStencilDescStatements() const {
    return stencilDescStatements_;
  }

  /// @brief Get the stencil description AST
  inline std::vector<std::shared_ptr<Statement>>& getStencilDescStatements() {
    return stencilDescStatements_;
  }

  /// @brief Get the list of stencil functions
  inline std::vector<std::shared_ptr<StencilFunctionInstantiation>>&
  getStencilFunctionInstantiations() {
    return stencilFunctionInstantiations_;
  }

  inline const std::vector<std::shared_ptr<StencilFunctionInstantiation>>&
  getStencilFunctionInstantiations() const {
    return stencilFunctionInstantiations_;
  }

  /// @brief Get map which associates Stmts with AccessIDs
  std::unordered_map<std::shared_ptr<Stmt>, int>& getStmtToAccessIDMap();
  const std::unordered_map<std::shared_ptr<Stmt>, int>& getStmtToAccessIDMap() const;

  /// @brief Get the AccessID-to-Name map
  std::unordered_map<std::string, int>& getNameToAccessIDMap();
  const std::unordered_map<std::string, int>& getNameToAccessIDMap() const;

  /// @brief Get the Name-to-AccessID map
  std::unordered_map<int, std::string>& getAccessIDToNameMap();
  const std::unordered_map<int, std::string>& getAccessIDToNameMap() const;

  /// @brief Get the Literal-AccessID-to-Name map
  std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap();
  const std::unordered_map<int, std::string>& getLiteralAccessIDToNameMap() const;

  /// @brief Get the StageID-to-Name map
  std::unordered_map<int, std::string>& getStageIDToNameMap();
  const std::unordered_map<int, std::string>& getStageIDToNameMap() const;

  /// @brief Get the field-AccessID set
  std::set<int>& getFieldAccessIDSet();
  const std::set<int>& getFieldAccessIDSet() const;

  /// @brief Get the field-AccessID set
  std::set<int>& getGlobalVariableAccessIDSet();
  const std::set<int>& getGlobalVariableAccessIDSet() const;

  inline const std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>&
  getBoundaryConditions() const {
    return FieldnameToBoundaryConditionMap_;
  }

  inline std::unordered_map<std::string, std::shared_ptr<BoundaryConditionDeclStmt>>&
  getBoundaryConditions() {
    return FieldnameToBoundaryConditionMap_;
  }

  inline const std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>&
  getBoundaryConditionToExtentsMap() const {
    return BoundaryConditionToExtentsMap_;
  }

  inline std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>&
  getBoundaryConditionToExtentsMap() {
    return BoundaryConditionToExtentsMap_;
  }

  SourceLocation& getStencilLocation();

  std::unordered_map<std::shared_ptr<Expr>, int>& getExprToAccessIDMap();

  std::set<int>& getTemporaryFieldAccessIDSet();

  std::set<int>& getAllocatedFieldAccessIDSet();

  VariableVersions& getVariableVersions();

  std::string getFileName();
  std::string getOriginalNameFromAccessID(int AccessID, const IIR *iir) const;
};
} // namespace iir
} // namespace dawn

#endif
