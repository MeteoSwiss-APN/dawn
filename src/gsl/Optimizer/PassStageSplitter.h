//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_OPTIMIZER_PASSSTAGESPLITTER_H
#define GSL_OPTIMIZER_PASSSTAGESPLITTER_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

/// @brief Pass for splitting stages due to horizontal non-pointwiese read-before-write data
/// dependencies
///
/// @see hasHorizontalReadBeforeWriteConflict
/// @ingroup optimizer
class PassStageSplitter : public Pass {
public:
  PassStageSplitter();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;
};

} // namespace gsl

#endif
