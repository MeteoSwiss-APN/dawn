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

#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/Array.h"
#include <algorithm>

namespace dawn {

PassSetBlockSize::PassSetBlockSize(OptimizerContext& context) : Pass(context, "PassSetBlockSize") {}

bool PassSetBlockSize::run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  const auto& IIR = stencilInstantiation->getIIR();

  Array3ui blockSize{static_cast<unsigned int>(context_.getOptions().block_size_i),
                     static_cast<unsigned int>(context_.getOptions().block_size_j),
                     static_cast<unsigned int>(context_.getOptions().block_size_k)};
  if(std::all_of(blockSize.begin(), blockSize.end(), [](unsigned int size) { return size == 0; })) {
    bool verticalPattern = true;
    for(const auto& stage : iterateIIROver<iir::Stage>(*IIR)) {
      if(!stage->getExtents().isHorizontalPointwise()) {
        verticalPattern = false;
      }
    }
    for(const auto& stencil : (*IIR).getChildren()) {
      for(const auto& fieldP : stencil->getFields()) {
        const auto& field = fieldP.second;

        auto extent = field.field.getExtentsRB();
        auto const& hExtent =
            dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extent.horizontalExtent());

        if(hExtent.jPlus() != 0 || hExtent.jMinus() != 0) {
          verticalPattern = false;
        }
      }
    }

    // recent generation of GPU architectures show good memory bandwidth with <32,1> block sizes,
    // but if there are horizontal data dependencies, the redundant accesses across different blocks
    // limit the performance
    if(verticalPattern) {
      blockSize = {32, 1, 4};
    } else {
      blockSize = {32, 4, 4};
    }
  }

  IIR->setBlockSize(blockSize);

  if(context_.getOptions().ReportPassSetBlockSize) {
    std::cout << "\nPASS: " << getName() << ": " << stencilInstantiation->getName() << ": blockSize"
              << "[" << blockSize[0] << "," << blockSize[1] << "," << blockSize[2] << "]"
              << std::endl;
  }

  // Notice that gridtools does not supported yet setting different block sizes, therefore the block
  // size of the IIR is currently ignored by GT

  return true;
}

} // namespace dawn
