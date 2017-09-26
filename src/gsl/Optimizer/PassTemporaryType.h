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

#ifndef GSL_OPTIMIZER_PASSTEMPORARYTYPE_H
#define GSL_OPTIMIZER_PASSTEMPORARYTYPE_H

#include "gsl/Optimizer/Pass.h"

namespace gsl {

class Stencil;

/// @brief Set the correct type of the temporaries
///
/// Temporaries can be either fields (i.e storages) or local variables. This pass checks the type
/// of each temporary and adjust it if necessary.
///
/// During the splitting passes we may break references to local variables. Consider the following
/// example:
///
/// @b Stage
/// @code
///   double some_variable = 5.0;
///   foo = input
///   output = foo(i+1) + some_variable;
/// @endcode
///
/// We will split this into two stages (due to the horizontal dependency)
///
/// @b Stage (1)
/// @code
///   double some_variable = 5.0;
///   foo = input
/// @endcode
///
/// @b Stage (2)
/// @code
///   output = foo(i+1) + some_variable;
/// @endcode
///
/// which breaks the reference to `some_variable` (in gridtools stages are separate `structs`).
/// The solution is thus to promote `some_variable` to a temporary field. Note that the reverse can
/// also happen, we may have a temporary field which is only accessed within a single stage at no
/// offsets (most likely the result of inlining a nested stencil function), we can thus demote the
/// temporary field to a local variable which will decrease shared memory consumption.
///
/// @note
/// In order to leverage its full potential this pass should be ran after the splitting
/// passes and again after the reordering passes.
///
/// @ingroup optimizer
class PassTemporaryType : public Pass {
public:
  PassTemporaryType();

  /// @brief Pass implementation
  bool run(StencilInstantiation* stencilInstantiation) override;

  /// @brief Promote a temporary fields which span over multiple stencils to real (allocated)
  /// storage
  ///
  /// We check if a temporary field is referenced in more than one stencil and, if so, we promote
  /// the field to a real (manaully allocated) field.
  static void
  fixTemporariesSpanningMultipleStencils(StencilInstantiation* instantiation,
                                         const std::vector<std::shared_ptr<Stencil>>& stencils);
};

} // namespace gsl

#endif
