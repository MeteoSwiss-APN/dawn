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

#ifndef GSL_CODEGEN_GTCLANGCODEGEN_H
#define GSL_CODEGEN_GTCLANGCODEGEN_H

#include "gsl/CodeGen/CodeGen.h"
#include "gsl/Optimizer/Interval.h"
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace gsl {

class StencilInstantiation;
class OptimizerContext;
class Stage;
class Stencil;

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup codegen
class GTClangCodeGen : public CodeGen {
public:
  GTClangCodeGen(OptimizerContext* context);
  virtual ~GTClangCodeGen();

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
  std::string generateGlobals(const SIR* Sir);

  /// Maximum needed vector size of boost::fusion containers
  std::size_t mplContainerMaxSize_;
};

} // namespace gsl

#endif
