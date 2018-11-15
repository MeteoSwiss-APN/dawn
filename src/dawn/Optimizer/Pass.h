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

#include "dawn/Support/Assert.h"
#include <memory>
#include <string>
#include <unordered_map>
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
  std::unordered_map<std::string, bool> categorySet_;

public:
  Pass(const std::string& name) : Pass(name, false, true) {}
  Pass(const std::string& name, bool isDebug) : Pass(name, isDebug, true) {}
  Pass(const std::string& name, bool isDebug, bool afterSerialization) : name_(name) {
    categorySet_.insert({"Debug", isDebug});
    categorySet_.insert({"Deserialization", afterSerialization});
  }
  virtual ~Pass() {}

  /// @brief Run the the Pass
  /// @returns `true` on success, `false` otherwise
  virtual bool run(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) = 0;

  /// @brief Get the name of the Pass
  const std::string& getName() const { return name_; }

  /// @brief Get the dependencies of this pass
  std::vector<std::string> getDependencies() const { return dependencies_; }

  bool checkFlag(std::string flag) const {
    DAWN_ASSERT_MSG(categorySet_.count(flag), "unsupported flag");
    auto it = categorySet_.find(flag);
    return it->second;
  }
};

} // namespace dawn

#endif
