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

#ifndef DAWN_OPTIMIZER_PASSDATALOCALITYMETRIC_H
#define DAWN_OPTIMIZER_PASSDATALOCALITYMETRIC_H

#include "dawn/Optimizer/Pass.h"

namespace dawn {

struct HardwareConfig {
  /// Maximum number of fields concurrently in shared memory
  int SMemMaxFields = 8;

  /// Maximum number of fields concurrently in the texture cache
  int TexCacheMaxFields = 3;
};

/// @brief This Pass computes a heuristic measuring the data-locality of each stencil
///
/// @ingroup optimizer
class PassDataLocalityMetric : public Pass {
public:
  PassDataLocalityMetric();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;

private:
  HardwareConfig config_;
};

} // namespace dawn

#endif
