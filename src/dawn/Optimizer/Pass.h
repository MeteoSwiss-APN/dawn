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

#ifndef DAWN_OPTIMIZER_PASS_H
#define DAWN_OPTIMIZER_PASS_H

#include <memory>
#include <string>
#include <vector>

namespace dawn {

class StencilInstantiation;

/// @brief Abstract base class of all optimization and analyzer passes
///
/// To create a new Pass proceed as follows:
///
///   1) Inherit from `Pass` and implement the Pass::run method. Remeber to return `true`
///      on success!
///   2) Register your new Pass in the DAWNCompiler::compile method in DAWNCompiler.cpp at the
///      position you would like it run.
///
/// @ingroup Optimizer
class Pass {
  /// Name of the pass (should be unique)
  std::string name_;

protected:
  /// Name of the passes this pass depends on (empty implies no dependency)
  std::vector<std::string> dependencies_;
  /// Categroy of the pass
  bool isDebug_ = false;

public:
  Pass(const std::string& name) : name_(name) {}
  virtual ~Pass() {}

  /// @brief Run the the Pass
  /// @returns `true` on success, `false` otherwise
  virtual bool run(const std::shared_ptr<StencilInstantiation>& stencilInstantiation) = 0;

  /// @brief Get the name of the Pass
  const std::string& getName() const { return name_; }

  /// @brief Get the dependencies of this pass
  std::vector<std::string> getDependencies() const { return dependencies_; }

  bool isDebug() const { return isDebug_; }
};

} // namespace dawn

#endif
