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

#include "dawn/Optimizer/Pass.h"
#include "dawn/Support/Assert.h"
#include <set>
#include <unordered_map>

namespace dawn {

namespace iir {
class Stencil;
class DoMethod;
} // namespace iir

struct SkipIDs {
  std::unordered_map<int, std::set<int>> accessIDs;
  void insertAccessIDsOfMS(int MSID, std::set<int>&& ids) { accessIDs[MSID] = std::move(ids); }
  void appendAccessIDsToMS(int MSID, const int id) { accessIDs[MSID].insert(id); }
  bool skipID(const int MSID, const int id) const {
    DAWN_ASSERT(accessIDs.count(MSID));
    return accessIDs.at(MSID).count(id);
  }
};

/// @brief PassTemporaryToStencilFunction pass will identify temporaries of a stencil and replace
/// their pre-computations
/// by a stencil function. Each reference to the temporary is later replaced by the stencil function
/// call.
/// * Input: well formed SIR and IIR with the list of mss/stages, temporaries used and
/// statements with accesses already computed
/// * Output: modified SIR, new stencil functions are inserted and calls. Temporary fields are
/// removed. New stencil functions instantiations are inserted into the IIR. Statements' accesses
/// are recomputed
/// @ingroup optimizer
///
/// This pass is not necessary to create legal code and is hence not in the debug-group
class PassTemporaryToStencilFunction : public Pass {
public:
  PassTemporaryToStencilFunction(OptimizerContext& context);

  /// @brief Pass implementation
  bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) override;

private:
  SkipIDs computeSkipAccessIDs(
      const std::unique_ptr<iir::Stencil>& stencilPtr,
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const;
};

} // namespace dawn
