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

#ifndef DAWN_IIR_FIELDACCESSMETADATA_H
#define DAWN_IIR_FIELDACCESSMETADATA_H

#include "dawn/Support/Json.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {
namespace iir {

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

enum class FieldAccessType {
  FAT_GlobalVariable, // a global variable (i.e. not field with grid dimensiontality)
  FAT_Literal,        // a literal that is not stored in memory
  FAT_MemoryField // a access to data tha resides in memory with a field dimensionality (i.e. not
                  // scalar value)
};

enum class FieldAccessScope {
  FAS_LocalVariable,
  FAS_StencilTemporary,
  FAS_InterStencilTemporary,
  FAS_Field
};

struct FieldAccessMetadata {

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

  std::set<int> AllocatedFieldAccessIDSet_;

  void clone(const FieldAccessMetadata& origin);
};
} // namespace iir
} // namespace dawn

#endif
