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

#include "dawn/Compiler/Options.h"
#include "dawn/IIR/MetaInformation.h"
#include "dawn/IIR/Stencil.h"

namespace dawn {
class OptimizerContext;
class DiagnosticsEngine;
class HardwareConfig;

namespace iir {

/// @brief A Stencil is represented by a collection of MultiStages
/// @ingroup optimizer
class IIR : public IIRNode<void, IIR, Stencil> {

  std::shared_ptr<StencilMetaInformation> metadata_;

  struct DerivedInfo {};

  DerivedInfo derivedInfo_;

  OptimizerContext* creator_;

public:
  static constexpr const char* name = "IIR";

  using StencilSmartPtr_t = child_smartptr_t<Stencil>;

  /// @brief constructors and assignment
  IIR(); //= default;
  IIR(OptimizerContext* creator);
  IIR(const IIR&) = default;
  IIR(IIR&&) = default;
  IIR& operator=(const IIR&) = default;
  IIR& operator=(IIR&&) = default;
  /// @}
  /// @brief clone the IIR
  std::unique_ptr<IIR> clone() const;

  /// @brief update the derived info from children
  virtual void updateFromChildren() override;

  std::shared_ptr<StencilMetaInformation> getMetaData() { return metadata_; }
  const std::shared_ptr<StencilMetaInformation>& getMetaData() const { return metadata_; }

  Options& getOptions();

  const DiagnosticsEngine& getDiagnostics() const;
  DiagnosticsEngine& getDiagnostics();

  const HardwareConfig& getHardwareConfiguration() const;
  HardwareConfig& getHardwareConfiguration();

};
} // namespace iir
} // namespace dawn

#endif
