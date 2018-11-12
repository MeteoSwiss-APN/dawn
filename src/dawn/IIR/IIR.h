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

#ifndef DAWN_IIR_IIR_H
#define DAWN_IIR_IIR_H

#include "dawn/IIR/Stencil.h"
#include <set>

namespace dawn {
namespace iir {

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class IIR : public IIRNode<void, IIR, Stencil> {

  const std::array<unsigned int, 3> blockSize_ = {{32, 4, 4}};

  struct DerivedInfo {
    /// Can be filled from the AccessIDToName map that is in Metainformation
    std::unordered_map<std::string, int> NameToAccessIDMap_;

    /// Can be filled from the StencilIDToStencilCallMap that is in Metainformation
    std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int> StencilCallToStencilIDMap_;

    /// StageID to name Map. Filled by the `PassSetStageName`.
    std::unordered_map<int, std::string> StageIDToNameMap_;

    /// BoundaryConditionCall to Extent Map. Filled my `PassSetBoundaryCondition`
    std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>
        BoundaryConditionToExtentsMap_;

    /// Set containing the AccessIDs of fields which are manually allocated by the stencil and serve
    /// as temporaries spanning over multiple stencils
    std::set<int> AllocatedFieldAccessIDSet_;
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "IIR";

  using StencilSmartPtr_t = child_smartptr_t<Stencil>;

  inline std::array<unsigned int, 3> getBlockSize() const { return blockSize_; }

  /// @brief constructors and assignment
  IIR() = default;
  IIR(const IIR&) = default;
  IIR(IIR&&) = default;
  IIR& operator=(const IIR&) = default;
  IIR& operator=(IIR&&) = default;
  /// @}
  /// @brief clone the IIR
  std::unique_ptr<IIR> clone() const;

  /// @brief clone the IIR
  void clone(std::unique_ptr<IIR>& dest) const;
  }

  std::unordered_map<std::string, int>& getNameToAccessIDs() {
    return derivedInfo_.NameToAccessIDMap_;
  }

  std::unordered_map<int, std::string>& getStageIDToNameMap() {
    return derivedInfo_.StageIDToNameMap_;
  }
  const std::string& getNameFromStageID(int StageID) const {
    auto it = derivedInfo_.StageIDToNameMap_.find(StageID);
    DAWN_ASSERT_MSG(it != derivedInfo_.StageIDToNameMap_.end(), "Invalid StageID");
    return it->second;
  }

  std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>&
  getBoundaryConditionToExtents() {
    return derivedInfo_.BoundaryConditionToExtentsMap_;
  }

  std::unordered_map<std::shared_ptr<StencilCallDeclStmt>, int>& getStencilCallToStencilIDMap() {
    return derivedInfo_.StencilCallToStencilIDMap_;
  }
  std::set<int>& getAllocatedFieldAccessIDSet() { return derivedInfo_.AllocatedFieldAccessIDSet_; }
};
} // namespace iir
} // namespace dawn

#endif
