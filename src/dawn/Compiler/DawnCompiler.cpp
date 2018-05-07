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

#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassStencilSplitter.h"
#include "dawn/Optimizer/PassTemporaryFirstAccess.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/EditDistance.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

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

DawnCompiler::DawnCompiler(Options* options) : diagnostics_(make_unique<DiagnosticsEngine>()) {
  options_ = options ? make_unique<Options>(*options) : make_unique<Options>();
}

std::unique_ptr<OptimizerContext> DawnCompiler::runOptimizer(std::shared_ptr<SIR> const& SIR) {
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

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;
  MultistageSplitStrategy mssSplitStrategy;
  if(options_->MaxCutMSS) {
    mssSplitStrategy = MultistageSplitStrategy::SS_MaxCut;
  } else {
    mssSplitStrategy = MultistageSplitStrategy::SS_Optimized;
  }


  // -max-fields
  int maxFields = options_->MaxFieldsPerStencil;

  // Initialize optimizer
  std::unique_ptr<OptimizerContext> optimizer =
      make_unique<OptimizerContext>(getDiagnostics(), getOptions(), SIR);
  PassManager& passManager = optimizer->getPassManager();

  // Setup pass interface
  optimizer->checkAndPushBack<PassInlining>(inlineStrategy);
  optimizer->checkAndPushBack<PassTemporaryFirstAccess>();
  optimizer->checkAndPushBack<PassFieldVersioning>();
  optimizer->checkAndPushBack<PassSSA>();
  optimizer->checkAndPushBack<PassMultiStageSplitter>(mssSplitStrategy);
  optimizer->checkAndPushBack<PassStageSplitter>();
  optimizer->checkAndPushBack<PassPrintStencilGraph>();
  optimizer->checkAndPushBack<PassTemporaryType>();
  optimizer->checkAndPushBack<PassSetStageName>();
  optimizer->checkAndPushBack<PassSetStageGraph>();
  optimizer->checkAndPushBack<PassStageReordering>(reorderStrategy);
  optimizer->checkAndPushBack<PassStageMerger>();
  optimizer->checkAndPushBack<PassStencilSplitter>(maxFields);
  optimizer->checkAndPushBack<PassTemporaryType>();
  optimizer->checkAndPushBack<PassTemporaryMerger>();
  optimizer->checkAndPushBack<PassTemporaryToStencilFunction>();
  optimizer->checkAndPushBack<PassSetNonTempCaches>();
  optimizer->checkAndPushBack<PassSetCaches>();
  optimizer->checkAndPushBack<PassComputeStageExtents>();
  optimizer->checkAndPushBack<PassSetBoundaryCondition>();
  optimizer->checkAndPushBack<PassDataLocalityMetric>();

  DAWN_LOG(INFO) << "All the passes ran with the current command line arugments:";
  for(const auto& a : passManager.getPasses()) {
    DAWN_LOG(INFO) << a->getName();
  }

  // Run optimization passes
  for(auto& stencil : optimizer->getStencilInstantiationMap()) {
    std::shared_ptr<StencilInstantiation> instantiation = stencil.second;
    DAWN_LOG(INFO) << "Starting Optimization and Analysis passes for `" << instantiation->getName()
                   << "` ...";
    if(!passManager.runAllPassesOnStecilInstantiation(instantiation))
      return nullptr;
    DAWN_LOG(INFO) << "Done with Optimization and Analysis passes for `" << instantiation->getName()
                   << "`";
  }

  return optimizer;
}

std::unique_ptr<codegen::TranslationUnit> DawnCompiler::compile(const std::shared_ptr<SIR>& SIR,
                                                                CodeGenKind codeGen) {
  diagnostics_->clear();
  diagnostics_->setFilename(SIR->Filename);

  // Check if options are valid

  // -max-halo
  if(options_->MaxHaloPoints < 0) {
    diagnostics_->report(buildDiag("-max-halo", options_->MaxHaloPoints,
                                   "maximum number of allowed halo points must be >= 0"));
    return nullptr;
  }

  // Initialize optimizer
  auto optimizer = runOptimizer(SIR);

  if(diagnostics_->hasErrors()) {
    DAWN_LOG(INFO) << "Errors occured. Skipping code generation.";
    return nullptr;
  }

  // Generate code
  std::unique_ptr<codegen::CodeGen> CG;
  switch(codeGen) {
  case CodeGenKind::CG_GTClang:
    CG = make_unique<codegen::gt::GTCodeGen>(optimizer.get());
    break;
  case CodeGenKind::CG_GTClangNaiveCXX:
    CG = make_unique<codegen::cxxnaive::CXXNaiveCodeGen>(optimizer.get());
    break;
  case CodeGenKind::CG_GTClangOptCXX:
    dawn_unreachable("GTClangOptCXX not supported yet");
    break;
  }
  return CG->generateCode();
}

const DiagnosticsEngine& DawnCompiler::getDiagnostics() const { return *diagnostics_.get(); }
DiagnosticsEngine& DawnCompiler::getDiagnostics() { return *diagnostics_.get(); }

const Options& DawnCompiler::getOptions() const { return *options_.get(); }
Options& DawnCompiler::getOptions() { return *options_.get(); }

} // namespace dawn
