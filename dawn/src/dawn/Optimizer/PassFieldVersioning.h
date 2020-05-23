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

#include "dawn/IIR/LoopOrder.h"
#include "dawn/Optimizer/Pass.h"

namespace dawn {

namespace iir {
class Stencil;
class DependencyGraphAccesses;
class DoMethod;
} // namespace iir

/// @brief This pass resolves potential race condition by introducing double buffering i.e
/// versioning of fields
///
/// @see fixRaceCondition
/// @ingroup optimizer
class PassFieldVersioning : public Pass {

private:
  int numRenames_;

  /// @brief Kind of race condition encountered
  enum class RCKind {
    Nothing = 0, ///< No race-condition was encountered
    Fixed,       ///< Encounterd race-condition and fixed it by double buffering
    Unresolvable ///< Race condition cannot be fixed
  };

  /// @brief Fix race conditions by introducing double buffering of the offending fields
  ///
  /// Consider the following example:
  ///
  /// @code
  ///   tmp = u(i+1);
  ///   u = tmp;
  /// @endcode
  ///
  /// Due to our parallel model, we would have a race condition in `u`. Splitting the the two
  /// statements into different stages is insufficient, as there is no buffering of the
  /// boundary extents. The solution is to introduce a @b new field i.e double buffer the
  /// field `u`.
  ///
  /// @code
  ///   tmp = u_in(i+1);
  ///   u_out = tmp;
  /// @endcode
  ///
  /// This detects such patterns and, if applicable, fixes them. Note that not all race conditions
  /// can be resolved e.g race conditions inside stencil functions of block statements are currently
  /// not resolved and will produce a diagnostic message.
  ///
  /// @param graph      Dependency graph to analyze
  /// @param stencil    Current stencil
  /// @param stencil    Current Do-Method
  /// @param loopOrder  Current loop order of the stage
  /// @param stageIdx   @b Linear index of the stage in the stencil
  /// @param stmtIdx    Index of the statement inside the stage
  RCKind fixRaceCondition(const std::shared_ptr<iir::StencilInstantiation> instantiation,
                          const iir::DependencyGraphAccesses& graph, iir::Stencil& stencil,
                          iir::DoMethod& doMethod, iir::LoopOrderKind loopOrder, int stageIdx,
                          int stmtIdx);

public:
  PassFieldVersioning(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;
};

} // namespace dawn
