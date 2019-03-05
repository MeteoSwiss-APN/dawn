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

#include "dawn/IIR/ControlFlowDescriptor.h"
#include "dawn/IIR/Stencil.h"
#include <set>

namespace dawn {
namespace iir {

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class IIR : public IIRNode<void, IIR, Stencil> {

  std::array<unsigned int, 3> blockSize_ = {{32, 4, 4}};
  ControlFlowDescriptor controlFlowDesc_;

  struct DerivedInfo {
    /// StageID to name Map. Filled by the `PassSetStageName`.
    std::unordered_map<int, std::string> StageIDToNameMap_;
    /// Set containing the AccessIDs of fields which are manually allocated by the stencil and serve
    /// as temporaries spanning over multiple stencils
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

  json::json jsonDump() const;

  const ControlFlowDescriptor& getControlFlowDescriptor() const { return controlFlowDesc_; }
  // TODO do not have non const?
  ControlFlowDescriptor& getControlFlowDescriptor() { return controlFlowDesc_; }
  inline void setBlockSize(const std::array<unsigned int, 3> blockSize) { blockSize_ = blockSize; }

  inline std::unordered_map<int, std::string>& getStageIDToNameMap() {
    return derivedInfo_.StageIDToNameMap_;
  }
  inline const std::string& getNameFromStageID(int StageID) const {
    auto it = derivedInfo_.StageIDToNameMap_.find(StageID);
    DAWN_ASSERT_MSG(it != derivedInfo_.StageIDToNameMap_.end(), "Invalid StageID");
    return it->second;
  }

  //  inline std::unordered_map<std::shared_ptr<BoundaryConditionDeclStmt>, Extents>&
  //  getBoundaryConditionToExtents() {
  //    return derivedInfo_.BoundaryConditionToExtentsMap_;
  //  }
};
} // namespace iir
} // namespace dawn

#endif
