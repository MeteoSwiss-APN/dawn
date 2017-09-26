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

#ifndef GSL_OPTIMIZER_PASS_H
#define GSL_OPTIMIZER_PASS_H

#include <string>
#include <vector>

namespace gsl {

class StencilInstantiation;

/// @brief Abstract base class of all optimization and analyzer passes
///
/// To create a new Pass proceed as follows:
///
///   1) Inherit from `Pass` and implement the Pass::run method. Remeber to return `true`
///      on success!
///   2) Register your new Pass in the GSLCompiler::compile method in GSLCompiler.cpp at the
///      position you would like it run.
///
/// @ingroup Optimizer
class Pass {
  /// Name of the pass (should be unique)
  std::string name_;

protected:
  /// Name of the passes this pass depends on (empty implies no dependency)
  std::vector<std::string> dependencies_;

public:
  Pass(const std::string& name) : name_(name) {}
  virtual ~Pass() {}

  /// @brief Run the the Pass
  /// @returns `true` on success, `false` otherwise
  virtual bool run(StencilInstantiation* stencilInstantiation) = 0;

  /// @brief Get the name of the Pass
  const std::string& getName() const { return name_; }

  /// @brief Get the dependencies of this pass
  std::vector<std::string> getDependencies() const { return dependencies_; }
};

} // namespace gsl

#endif
