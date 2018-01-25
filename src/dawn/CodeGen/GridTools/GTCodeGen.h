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

#ifndef DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H
#define DAWN_CODEGEN_GRIDTOOLS_GTCODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace dawn {

class StencilInstantiation;
class OptimizerContext;
class Stage;
class Stencil;

namespace codegen {
namespace gt {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup gt
class GTCodeGen : public CodeGen {
public:
  GTCodeGen(OptimizerContext* context);
  virtual ~GTCodeGen();

  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  /// @brief Definitions of the gridtools::intervals
  struct IntervalDefinitions {
    IntervalDefinitions(const Stencil& stencil);

    /// Intervals of the stencil
    std::unordered_set<Interval> Intervals;

    /// Axis of the stencil (i.e the interval which spans accross all other intervals)
    Interval Axis;

    /// Levels of the axis
    std::set<int> Levels;

    /// Unqiue name of an interval
    std::unordered_map<Interval, std::string> IntervalToNameMap;

    /// Intervals of the Do-Methods of each stage
    std::unordered_map<std::shared_ptr<Stage>, std::vector<Interval>> StageIntervals;
  };

private:
  std::string generateStencilInstantiation(const StencilInstantiation* stencilInstantiation);
  std::string generateGlobals(const std::shared_ptr<SIR> Sir);

  /// Maximum needed vector size of boost::fusion containers
  std::size_t mplContainerMaxSize_;
};

} // namespace gt
} // namespace codegen
} // namespace dawn

#endif
