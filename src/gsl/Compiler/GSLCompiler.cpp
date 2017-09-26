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

#include "gsl/CodeGen/CodeGen.h"
#include "gsl/CodeGen/GTClangCodeGen.h"
#include "gsl/CodeGen/GTClangNaiveCXXCodeGen.h"
#include "gsl/Compiler/GSLCompiler.h"
#include "gsl/Optimizer/OptimizerContext.h"
#include "gsl/SIR/SIR.h"
#include "gsl/Support/EditDistance.h"
#include "gsl/Support/Logging.h"
#include "gsl/Support/StringSwitch.h"
#include "gsl/Support/StringUtil.h"

#include "gsl/Optimizer/PassDataLocalityMetric.h"
#include "gsl/Optimizer/PassFieldVersioning.h"
#include "gsl/Optimizer/PassInlining.h"
#include "gsl/Optimizer/PassMultiStageSplitter.h"
#include "gsl/Optimizer/PassPrintStencilGraph.h"
#include "gsl/Optimizer/PassSSA.h"
#include "gsl/Optimizer/PassSetCaches.h"
#include "gsl/Optimizer/PassSetStageGraph.h"
#include "gsl/Optimizer/PassSetStageName.h"
#include "gsl/Optimizer/PassStageMerger.h"
#include "gsl/Optimizer/PassStageReordering.h"
#include "gsl/Optimizer/PassStageSplitter.h"
#include "gsl/Optimizer/PassStencilSplitter.h"
#include "gsl/Optimizer/PassTemporaryFirstAccess.h"
#include "gsl/Optimizer/PassTemporaryMerger.h"
#include "gsl/Optimizer/PassTemporaryType.h"

namespace gsl {

namespace {

/// @brief Make a suggestion to the user if there is a small typo (only works with string options)
template <class T>
struct ComputeEditDistance {
  static std::string getSuggestion(const T& value, const std::vector<T>& possibleValues) {
    return "";
  }
};

template <>
struct ComputeEditDistance<std::string> {
  static std::string getSuggestion(const std::string& value,
                                   const std::vector<std::string>& possibleValues) {
    if(possibleValues.empty())
      return "";

    std::vector<unsigned> editDistances;
    std::transform(possibleValues.begin(), possibleValues.end(), std::back_inserter(editDistances),
                   [&](const std::string& val) { return computeEditDistance(value, val); });

    // Find minimum edit distance
    unsigned minEditDistance = editDistances[0], minEditDistanceIdx = 0;
    for(unsigned i = 1; i < editDistances.size(); ++i)
      if(editDistances[i] < minEditDistance) {
        minEditDistance = editDistances[i];
        minEditDistanceIdx = i;
      }

    return minEditDistance <= 2 ? "did you mean '" + possibleValues[minEditDistanceIdx] + "' ?"
                                : "";
  }
};

/// @brief Report a diagnostic concering an invalid Option
template <class T>
DiagnosticsBuilder buildDiag(const std::string& option, const T& value, std::string reason,
                             std::vector<T> possibleValues = std::vector<T>{}) {
  DiagnosticsBuilder diag(DiagnosticsKind::Error, SourceLocation());
  diag << "invalid value '" << value << "' of option '" << option << "'";

  if(!reason.empty()) {
    diag << ", " << reason;
  } else {
    auto suggestion = ComputeEditDistance<T>::getSuggestion(value, possibleValues);

    if(!suggestion.empty())
      diag << ", " << suggestion;
    else if(!possibleValues.empty())
      diag << ", possible values " << RangeToString()(possibleValues);
  }
  return diag;
}

} // anonymous namespace

GSLCompiler::GSLCompiler(Options* options) : diagnostics_(make_unique<DiagnosticsEngine>()) {
  options_ = options ? make_unique<Options>(*options) : make_unique<Options>();
}

std::unique_ptr<TranslationUnit> GSLCompiler::compile(const SIR* SIR, CodeGenKind codeGen) {
  diagnostics_->clear();
  diagnostics_->setFilename(SIR->Filename);

  // Check if options are valid

  // -max-halo
  if(options_->MaxHaloPoints < 0) {
    diagnostics_->report(buildDiag("-max-halo", options_->MaxHaloPoints,
                                   "maximum number of allowed halo points must be >= 0"));
    return nullptr;
  }

  // -inline
  using InlineStrategyKind = PassInlining::InlineStrategyKind;
  InlineStrategyKind inlineStrategy = StringSwitch<InlineStrategyKind>(options_->InlineStrategy)
                                          .Case("none", InlineStrategyKind::IK_None)
                                          .Case("cof", InlineStrategyKind::IK_ComputationOnTheFly)
                                          .Case("pc", InlineStrategyKind::IK_Precomputation)
                                          .Default(InlineStrategyKind::IK_Unknown);

  if(inlineStrategy == InlineStrategyKind::IK_Unknown) {
    diagnostics_->report(buildDiag("-inline", options_->InlineStrategy, "", {"none", "cof", "pc"}));
    return nullptr;
  }

  // -reorder
  using ReorderStrategyKind = ReorderStrategy::ReorderStrategyKind;
  ReorderStrategyKind reorderStrategy = StringSwitch<ReorderStrategyKind>(options_->ReorderStrategy)
                                            .Case("none", ReorderStrategyKind::RK_None)
                                            .Case("greedy", ReorderStrategyKind::RK_Greedy)
                                            .Case("scut", ReorderStrategyKind::RK_Partitioning)
                                            .Default(ReorderStrategyKind::RK_Unknown);

  if(reorderStrategy == ReorderStrategyKind::RK_Unknown) {
    diagnostics_->report(
        buildDiag("-reorder", options_->ReorderStrategy, "", {"none", "greedy", "scut"}));
    return nullptr;
  }

  // Initialize optimizer
  auto optimizer = make_unique<OptimizerContext>(this, SIR);
  PassManager& passManager = optimizer->getPassManager();

  // Setup pass interface
  passManager.pushBackPass<PassInlining>(inlineStrategy);
  passManager.pushBackPass<PassTemporaryFirstAccess>();
  passManager.pushBackPass<PassFieldVersioning>();
  passManager.pushBackPass<PassSSA>();
  passManager.pushBackPass<PassMultiStageSplitter>();
  passManager.pushBackPass<PassStageSplitter>();
  passManager.pushBackPass<PassPrintStencilGraph>();
  passManager.pushBackPass<PassTemporaryType>();
  passManager.pushBackPass<PassSetStageName>();
  passManager.pushBackPass<PassSetStageGraph>();
  passManager.pushBackPass<PassStageReordering>(reorderStrategy);
  passManager.pushBackPass<PassStageMerger>();
  passManager.pushBackPass<PassStencilSplitter>();
  passManager.pushBackPass<PassTemporaryType>();
  passManager.pushBackPass<PassTemporaryMerger>();
  passManager.pushBackPass<PassSetCaches>();
  passManager.pushBackPass<PassDataLocalityMetric>();

  // Run optimization passes
  for(auto& stencil : optimizer->getStencilInstantiationMap()) {
    StencilInstantiation* instantiation = stencil.second.get();
    GSL_LOG(INFO) << "Starting Optimization and Analysis passes for `" << instantiation->getName()
                  << "` ...";

    if(!passManager.runAllPassesOnStecilInstantiation(instantiation))
      return nullptr;

    GSL_LOG(INFO) << "Done with Optimization and Analysis passes for `" << instantiation->getName()
                  << "`";
  }

  if(diagnostics_->hasErrors()) {
    GSL_LOG(INFO) << "Errors occured. Skipping code generation.";
    return nullptr;
  }

  // Generate code
  std::unique_ptr<CodeGen> CG;
  switch(codeGen) {
  case CodeGenKind::CG_GTClang:
    CG = make_unique<GTClangCodeGen>(optimizer.get());
    break;
  case CodeGenKind::CG_GTClangNaiveCXX:
    CG = make_unique<GTClangNaiveCXXCodeGen>(optimizer.get());
    break;
  }
  return CG->generateCode();
}

const DiagnosticsEngine& GSLCompiler::getDiagnostics() const { return *diagnostics_.get(); }
DiagnosticsEngine& GSLCompiler::getDiagnostics() { return *diagnostics_.get(); }

const Options& GSLCompiler::getOptions() const { return *options_.get(); }
Options& GSLCompiler::getOptions() { return *options_.get(); }

} // namespace gsl
