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
  class VariableVersions {
  private:
    /// Map of AccessIDs to the the list of all AccessIDs of the multi-versioned variables. Note
    /// that the index in the vector corresponds to the version number.
    std::unordered_map<int, std::shared_ptr<std::vector<int>>> variableVersionsMap_;
    std::unordered_map<int, int> versionToOriginalVersionMap_;
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
    const std::unordered_set<int>& getVersionIDs() const { return versionIDs_; }
    VariableVersions() = default;
  };

public:
  ///@brief struct with properties of a stencil function instantiation candidate
  struct StencilFunctionInstantiationCandidate {
    /// stencil function instantiation from where the stencil function instantiation candidate is
    /// called
    std::shared_ptr<StencilFunctionInstantiation> callerStencilFunction_;
  };

  //================================================================================================
  // Stored MetaInformation
  //================================================================================================
  /// Map of AccessIDs and to the name of the variable/field. Note that only for fields of the "main
  /// stencil" we can get the AccessID by name. This is due the fact that fields of different
  /// stencil functions can share the same name.
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

  /// Set containing the AccessIDs of "global variable" accesses. Global variable accesses are
  /// represented by global_accessor or if we know the value at compile time we do a constant
  /// folding of the variable
  std::set<int> GlobalVariableAccessIDSet_;

  /// Map of AccessIDs to the list of all AccessIDs of the multi-versioned field, for fields and
  /// variables
  VariableVersions variableVersions_;

  /// Stencil description statements. These are built from the StencilDescAst of the sir::Stencil
  std::vector<std::shared_ptr<Statement>> stencilDescStatements_;
  std::unordered_map<int, std::shared_ptr<StencilCallDeclStmt>> IDToStencilCallMap_;

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

  void clone(const StencilMetaInformation& origin);
};
} // namespace iir
} // namespace dawn

#endif
