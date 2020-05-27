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

#pragma once

#include "dawn/AST/GridType.h"
#include "dawn/IIR/ControlFlowDescriptor.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/Support/RemoveIf.hpp"
#include <set>

namespace dawn {
namespace iir {

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class IIR : public IIRNode<void, IIR, Stencil> {

  const ast::GridType gridType_;

  std::array<unsigned int, 3> blockSize_ = {{32, 4, 4}};
  ControlFlowDescriptor controlFlowDesc_;

  std::shared_ptr<sir::GlobalVariableMap> globalVariableMap_;
  std::vector<std::shared_ptr<sir::StencilFunction>> stencilFunctions_;

  struct DerivedInfo {
    /// StageID to name Map. Filled by the `PassSetStageName`.
    std::unordered_map<int, std::string> StageIDToNameMap_;
    /// field info properties
    std::unordered_map<int, Stencil::FieldInfo> fields_;
    void clear();
  };

  DerivedInfo derivedInfo_;

public:
  static constexpr const char* name = "IIR";

  using StencilSmartPtr_t = child_smartptr_t<Stencil>;

  ast::GridType getGridType() const { return gridType_; }

  inline std::array<unsigned int, 3> getBlockSize() const { return blockSize_; }

  /// @brief constructors and assignment
  IIR(const ast::GridType gridType, std::shared_ptr<sir::GlobalVariableMap> sirGlobals,
      const std::vector<std::shared_ptr<sir::StencilFunction>>& stencilFunction);
  IIR(const IIR&) = default;
  IIR(IIR&&) = default;
  /// @}

  /// @brief clone the IIR
  std::unique_ptr<IIR> clone() const;
  /// @brief clone the IIR
  void clone(std::unique_ptr<IIR>& dest) const;

  json::json jsonDump() const;

  /// @brief update the derived info from children
  virtual void updateFromChildren() override;

  /// @brief returns true if the accessid is used within the stencil
  bool hasFieldAccessID(const int accessID) const { return derivedInfo_.fields_.count(accessID); }

  /// @brief Get the pair <AccessID, field> for the fields used within the multi-stage
  const std::unordered_map<int, Stencil::FieldInfo>& getFields() const {
    return derivedInfo_.fields_;
  }

  const std::vector<std::shared_ptr<sir::StencilFunction>>& getStencilFunctions() {
    return stencilFunctions_;
  }
  const ControlFlowDescriptor& getControlFlowDescriptor() const { return controlFlowDesc_; }
  ControlFlowDescriptor& getControlFlowDescriptor() { return controlFlowDesc_; }
  inline void setBlockSize(const std::array<unsigned int, 3> blockSize) { blockSize_ = blockSize; }

  inline std::unordered_map<int, std::string>& getStageIDToNameMap() {
    return derivedInfo_.StageIDToNameMap_;
  }
  void insertStencilFunction(const std::shared_ptr<sir::StencilFunction>& sirStencilFunction) {
    stencilFunctions_.push_back(sirStencilFunction);
  }

  inline const std::string& getNameFromStageID(int StageID) const {
    auto it = derivedInfo_.StageIDToNameMap_.find(StageID);
    DAWN_ASSERT_MSG(it != derivedInfo_.StageIDToNameMap_.end(), "Invalid StageID");
    return it->second;
  }

  std::shared_ptr<sir::GlobalVariableMap> getGlobalVariableMapPtr() const {
    return globalVariableMap_;
  }
  const sir::GlobalVariableMap& getGlobalVariableMap() const { return *globalVariableMap_; }

  void insertGlobalVariable(const std::string& varName, sir::Global&& value) {
    globalVariableMap_->insert(std::pair(varName, std::move(value)));
  }

  const Stencil& getStencil(const int stencilID) const;
};
} // namespace iir
} // namespace dawn
