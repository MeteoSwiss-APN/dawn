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

#ifndef DAWN_IIR_STENCILFUNCTION_H
#define DAWN_IIR_STENCILFUNCTION_H

#include "dawn/IIR/AST.h"
#include "dawn/Support/SourceLocation.h"
#include <memory>
#include <vector>

namespace dawn {

namespace sir {
struct StencilFunctionArg;
struct Interval;
struct StencilFunction;
} // namespace sir

namespace iir {

/// @brief IIR Stencil function
/// @ingroup iir
struct StencilFunction {
  std::string Name;
  SourceLocation Loc;
  std::vector<std::shared_ptr<sir::StencilFunctionArg>> Args;
  std::vector<std::shared_ptr<sir::Interval>> Intervals;
  std::vector<std::shared_ptr<iir::AST>> Asts;

  StencilFunction() {}
  StencilFunction(const sir::StencilFunction&);

  bool isSpecialized() const { return !Intervals.empty(); }

  /// @brief Get the AST of the specified vertical interval or `NULL` if the function is not
  /// specialized for this interval
  std::shared_ptr<iir::AST> getASTOfInterval(const sir::Interval& interval) const;

  bool hasArg(std::string name);
};

} // namespace iir

} // namespace dawn

#endif
