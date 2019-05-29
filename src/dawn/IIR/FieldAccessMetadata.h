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

#include "boost/variant.hpp"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Json.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {
namespace iir {

class VariableVersions {
public:
  VariableVersions() = default;

private:
  /// This map links the original fieldID with a list of all it's versioned fields. The index of
  /// the field in the vector denotes the version of the field
  std::unordered_map<int, std::shared_ptr<std::vector<int>>> variableVersionsMap_;

  struct DerivedInfo {
    /// This map links all the versions of a field to their original field. Can be derived by
    /// looping the variable-version map.
    std::unordered_map<int, int> versionToOriginalVersionMap_;
    /// This set contrains all the Fields that are versions of an original variable (excluding the
    /// originals). This is derived as it is the collection of keys in
    /// versionToOriginalVersionMap_
    std::unordered_set<int> versionIDs_;
  };
  DerivedInfo derivedInfo_;

public:
  bool variableHasMultipleVersions(const int accessID) const {
    return variableVersionsMap_.count(accessID);
  }

  std::shared_ptr<std::vector<int>> getVersions(const int accessID) const {
    return variableVersionsMap_.at(accessID);
  }

  void insertIDPair(const int originalAccessID, const int versionedAccessID) {
    // Insert the versioned ID into the list of verisons for its origial field
    if(variableHasMultipleVersions(originalAccessID)) {
      variableVersionsMap_[originalAccessID]->push_back(versionedAccessID);
    } else {
      variableVersionsMap_[originalAccessID] =
          std::make_shared<std::vector<int>>(1, versionedAccessID);
    }
    // Insert the versioned ID into the list of all versioned fields
    derivedInfo_.versionIDs_.insert(versionedAccessID);
    // and map it to it's origin
    derivedInfo_.versionToOriginalVersionMap_[versionedAccessID] = originalAccessID;
  }

  void removeID(const int accessID) {
    if(derivedInfo_.versionIDs_.count(accessID) > 0) {
      // Remove the field from the versions of it's original field
      int originalID = getOriginalVersionOfAccessID(accessID);
      auto vec = variableVersionsMap_[originalID];
      std::remove_if(vec->begin(), vec->end(), [&accessID](int id) { return id == accessID; });
      // Remove the backward map to it's origin
      derivedInfo_.versionToOriginalVersionMap_.erase(accessID);
      // and clear it from the list of all versioned fields
      derivedInfo_.versionIDs_.erase(accessID);
    }
  }

  bool isAccessIDAVersion(const int accessID) const {
    return derivedInfo_.versionIDs_.count(accessID);
  }

  int getOriginalVersionOfAccessID(const int accessID) const {
    if(isAccessIDAVersion(accessID)) {
      return derivedInfo_.versionToOriginalVersionMap_.at(accessID);
    } else {
      DAWN_ASSERT_MSG(0, "try to access original version of non-versioned field");
    }
    return 0;
  }

  const std::unordered_set<int>& getVersionIDs() const { return derivedInfo_.versionIDs_; }

  const std::unordered_map<int, std::shared_ptr<std::vector<int>>>& getvariableVersionsMap() const {
    return variableVersionsMap_;
  }

  json::json jsonDump() const;

  // now that we only have getters and setters this should always be in a consistent state.
  // Should we remove this enirely, have it here as a safety valve if one wants to mess with the
  // versioning and needs a clean update or have this as part of the update-level step for the
  // top-most level holding the metadata?
  // I personally think option 1 should be desired as this is (hopefully) never necessary but I've
  // put it here if the reviewers think this would be helpful
  void recomputeDerivedInfo() {
    derivedInfo_.versionIDs_.clear();
    derivedInfo_.versionToOriginalVersionMap_.clear();
    for(auto IDToVersionsPair : variableVersionsMap_) {
      int originalID = IDToVersionsPair.first;
      std::shared_ptr<std::vector<int>> listOfVersions = IDToVersionsPair.second;
      for(auto versionID : *listOfVersions) {
        derivedInfo_.versionIDs_.insert(versionID);
        derivedInfo_.versionToOriginalVersionMap_[versionID] = originalID;
      }
    }
  }
};

enum class FieldAccessType {
  FAT_GlobalVariable, // a global variable (i.e. not field with grid dimensiontality)
  FAT_Literal,        // a literal that is not stored in memory
  FAT_LocalVariable,
  FAT_StencilTemporary,
  FAT_InterStencilTemporary,
  FAT_Field,
  FAT_APIField
};

std::string toString(FieldAccessType type);

namespace impl {
// Needed for some older versions of GCC
template <typename...>
struct voider {
  using type = void;
};

// std::void_t will be part of C++17, but until then define it ourselves:
template <typename... T>
using void_t = typename voider<T...>::type;

template <typename T, typename U = void>
struct is_mapp_impl : std::false_type {};

template <typename T>
struct is_mapp_impl<
    T, void_t<typename T::key_type, typename T::mapped_type,
              decltype(std::declval<T&>()[std::declval<const typename T::key_type&>()])>>
    : std::true_type {
  void t() { T::kk(); }
};
}

template <FieldAccessType TFieldAccessType>
struct TypeOfAccessContainer;

template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_GlobalVariable> {
  using type = std::set<int>;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_Literal> {
  using type = std::unordered_map<int, std::string>;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_LocalVariable> {
  using type = void;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_StencilTemporary> {
  using type = std::set<int>;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_InterStencilTemporary> {
  using type = std::set<int>;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_Field> {
  using type = std::set<int>;
};
template <>
struct TypeOfAccessContainer<FieldAccessType::FAT_APIField> {
  using type = std::vector<int>;
};

template <typename T>
struct key_of {
  using type = typename T::key_type;
};
template <typename T>
struct value_of {
  using type = typename T::value_type;
};

template <FieldAccessType TFieldAccessType>
struct AccessesContainerKeyValue {
  using key_t = typename std::conditional<
      impl::is_mapp_impl<typename TypeOfAccessContainer<TFieldAccessType>::type>::value,
      key_of<typename TypeOfAccessContainer<TFieldAccessType>::type>,
      value_of<typename TypeOfAccessContainer<TFieldAccessType>::type>>::type;
  using value_t = typename TypeOfAccessContainer<TFieldAccessType>::type::value_type;
};

struct FieldAccessMetadata {

  using allContainerTypes =
      boost::variant<std::unordered_map<int, std::string>&, std::set<int>&, std::vector<int>&>;

  using allConstContainerTypes = boost::variant<const std::unordered_map<int, std::string>&,
                                                const std::set<int>&, const std::vector<int>&>;
  // Rules:
  // - FieldAccessIDSet_ includes : apiFieldIDs_, TemporaryFieldAccessIDSet_,
  // AllocatedFieldAccessIDSet_
  // - GlobalVariableAccessIDSet_, LiteralAccessIDToNameMap_, FieldAccessIDSet_ are exclusive

  /// Injection of AccessIDs of literal constant to their respective name (usually the name is
  /// just the string representation of the value). Note that literals always have *strictly*
  /// negative AccessIDs, which makes them distinguishable from field or variable AccessIDs. Further
  /// keep in mind that each access to a literal creates a new AccessID!
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

  std::unordered_map<int, FieldAccessType> accessIDType_;

  void clone(const FieldAccessMetadata& origin);
};
} // namespace iir
} // namespace dawn

#endif
