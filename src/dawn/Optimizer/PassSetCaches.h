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

#ifndef DAWN_OPTIMIZER_PASSSETMULTISTAGECACHES_H
#define DAWN_OPTIMIZER_PASSSETMULTISTAGECACHES_H

#include "dawn/IIR/Cache.h"
#include "dawn/IIR/Interval.h"
#include "dawn/IIR/Stage.h"
#include "dawn/IIR/Stencil.h"
#include "dawn/Optimizer/Pass.h"

namespace dawn {

/// @brief Determine which fields can be cached during the executation of the multi-stage
///
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassSetCaches : public Pass {
public:
  enum CachingStrategy {
    CS_MaximizeCaches,  ///< Sets Caches
    CS_GeneticAlgorithm ///< Runs on a Working IIR and does mating and mutation
  };

  PassSetCaches(CachingStrategy strategy);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;

private:
  void setAllCaches(const std::shared_ptr<iir::StencilInstantiation>& instantiation);

  void geneticAlgorithm(const std::shared_ptr<iir::StencilInstantiation>& instantiation);

  CachingStrategy strategy_;
};

} // namespace dawn

#endif
