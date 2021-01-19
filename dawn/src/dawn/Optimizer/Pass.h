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

#include "dawn/Optimizer/Options.h"

#include <memory>
#include <string>
#include <vector>

namespace dawn {

namespace iir {
class StencilInstantiation;
}

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
  const bool isDebug_;

public:
  Pass(const std::string& name, bool isDebug = false) : name_(name), isDebug_(isDebug) {}
  virtual ~Pass() {}

  /// @brief Run the the Pass
  /// @returns `true` on success, `false` otherwise
  virtual bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                   const Options& options) = 0;

  /// @brief Get the name of the Pass
  const std::string& getName() const { return name_; }

  /// @brief Get the dependencies of this pass
  std::vector<std::string> getDependencies() const { return dependencies_; }

  bool isDebug() const { return isDebug_; }
};

} // namespace dawn
