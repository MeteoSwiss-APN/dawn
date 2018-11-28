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
public:
  enum PassGroup {
    PG_Optimizer,    ///< Passes that do Optimization but are not required to generate legal code
    PG_CodeLegality, ///< Passes that need to run to generate valid code
    PG_LegalAndOpti, ///< Passes that are required to run to generate legal code but have modes that
                     /// are used for optimization only
    PG_Diganostics   ///< Passes that are only required for diagnostics
  };

  Pass(const std::string& name, PassGroup group) : name_(name) {
    switch(group) {
    case PG_Optimizer:
      categorySet_.insert({"Debug", false});
      categorySet_.insert({"Deserialization", false});
      categorySet_.insert({"GeneticAlgorithm", false});
      break;
    case PG_CodeLegality:
      categorySet_.insert({"Debug", true});
      categorySet_.insert({"Deserialization", true});
      categorySet_.insert({"GeneticAlgorithm", true});
      break;
    case PG_LegalAndOpti:
      categorySet_.insert({"Debug", true});
      categorySet_.insert({"Deserialization", false});
      categorySet_.insert({"GeneticAlgorithm", true});
      break;
    case PG_Diganostics:
      categorySet_.insert({"Debug", true});
      categorySet_.insert({"Deserialization", false});
      categorySet_.insert({"GeneticAlgorithm", false});
      break;
    }
  }
  Pass(const std::string& name, PassGroup group, bool isEnabled) : Pass(name, group) {
    categorySet_.insert({"MaunallyEnabled", isEnabled});
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

  int isManuallySwitched() const {
    if(categorySet_.count("MaunallyEnabled")) {
      auto it = categorySet_.find("MaunallyEnabled");
      return it->second;
    }
    return -1;
  }

private:
  /// Name of the pass (should be unique)
  std::string name_;

protected:
  /// Name of the passes this pass depends on (empty implies no dependency)
  std::vector<std::string> dependencies_;
  /// Categroy of the pass
  std::unordered_map<std::string, bool> categorySet_;
};

} // namespace dawn

#endif
