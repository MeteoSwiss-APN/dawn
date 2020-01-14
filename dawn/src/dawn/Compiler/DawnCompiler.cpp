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
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassStencilSplitter.h"
#include "dawn/Optimizer/PassTemporaryFirstAccess.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/EditDistance.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"
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
} // anonymous namespace

/// @brief Report a diagnostic concering an invalid Option
template <class T>
static DiagnosticsBuilder buildDiag(const std::string& option, const T& value, std::string reason,
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

static std::string remove_fileextension(std::string fullName, std::string extension) {
  std::string truncation = "";
  std::size_t pos = 0;
  while((pos = fullName.find(extension)) != std::string::npos) {
    truncation += fullName.substr(0, pos);
    fullName.erase(0, pos + extension.length());
  }
  return truncation;
}

static OptimizerContext::OptimizerContextOptions
createOptimizerOptionsFromAllOptions(const Options& options) {
  OptimizerContext::OptimizerContextOptions retval;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  retval.NAME = options.NAME;
#include "dawn/Optimizer/OptimizerOptions.inc"
#undef OPT
  return retval;
}

DawnCompiler::DawnCompiler() : options_(), diagnostics_(), filename_() {}

DawnCompiler::DawnCompiler(Options const& options)
    : options_(options), diagnostics_(), filename_() {}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
DawnCompiler::parallelize(std::shared_ptr<SIR> const& stencilIR) {
  diagnostics_.clear();
  diagnostics_.setFilename(stencilIR->Filename);

  // -reorder
  using ReorderStrategyKind = ReorderStrategy::Kind;
  ReorderStrategyKind reorderStrategy = StringSwitch<ReorderStrategyKind>(options_.ReorderStrategy)
                                            .Case("none", ReorderStrategyKind::None)
                                            .Case("greedy", ReorderStrategyKind::Greedy)
                                            .Case("scut", ReorderStrategyKind::Partitioning)
                                            .Default(ReorderStrategyKind::Unknown);

  if(reorderStrategy == ReorderStrategyKind::Unknown) {
    diagnostics_.report(
        buildDiag("-reorder", options_.ReorderStrategy, "", {"none", "greedy", "scut"}));
    throw std::runtime_error("An error occurred.");
  }

  IIRSerializer::Format serializationKind = IIRSerializer::Format::Json;
  if(options_.SerializeIIR || (options_.DeserializeIIR != "")) {
    if(options_.IIRFormat == "json") {
      serializationKind = IIRSerializer::Format::Json;
    } else if(options_.IIRFormat == "byte") {
      serializationKind = IIRSerializer::Format::Byte;
    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;
  MultistageSplitStrategy mssSplitStrategy;
  if(options_.MaxCutMSS) {
    mssSplitStrategy = MultistageSplitStrategy::MaxCut;
  } else {
    mssSplitStrategy = MultistageSplitStrategy::Optimized;
  }

  // Initialize optimizer
  OptimizerContext optimizer(getDiagnostics(), createOptimizerOptionsFromAllOptions(options_),
                             stencilIR);

  // required passes to have proper, parallelized IR
  optimizer.checkAndPushBack<PassInlining>(true, PassInlining::InlineStrategy::InlineProcedures);
  optimizer.checkAndPushBack<PassFieldVersioning>();
  optimizer.checkAndPushBack<PassMultiStageSplitter>(mssSplitStrategy);
  optimizer.checkAndPushBack<PassStageSplitter>();
  optimizer.checkAndPushBack<PassTemporaryType>();
  optimizer.checkAndPushBack<PassFixVersionedInputFields>();
  optimizer.checkAndPushBack<PassComputeStageExtents>();
  optimizer.checkAndPushBack<PassSetSyncStage>();

  DAWN_LOG(INFO) << "All the passes ran with the current command line arguments:";
  for(const auto& a : optimizer.getPassManager().getPasses()) {
    DAWN_LOG(INFO) << a->getName();
  }

  for(auto& stencil : optimizer.getStencilInstantiationMap()) {
    // Run optimization passes
    std::shared_ptr<iir::StencilInstantiation> instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting parallelization passes for `" << instantiation->getName()
                   << "` ...";
    if(!optimizer.getPassManager().runAllPassesOnStencilInstantiation(optimizer, instantiation))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with parallelization passes for `" << instantiation->getName() << "`";
  }

  return optimizer.getStencilInstantiationMap();
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
DawnCompiler::optimize(std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> const&
                           stencilInstantiationMap) {
  // -reorder
  using ReorderStrategyKind = ReorderStrategy::Kind;
  ReorderStrategyKind reorderStrategy = StringSwitch<ReorderStrategyKind>(options_.ReorderStrategy)
                                            .Case("none", ReorderStrategyKind::None)
                                            .Case("greedy", ReorderStrategyKind::Greedy)
                                            .Case("scut", ReorderStrategyKind::Partitioning)
                                            .Default(ReorderStrategyKind::Unknown);

  if(reorderStrategy == ReorderStrategyKind::Unknown) {
    diagnostics_.report(
        buildDiag("-reorder", options_.ReorderStrategy, "", {"none", "greedy", "scut"}));
    throw std::runtime_error("An error occurred.");
  }

  IIRSerializer::Format serializationKind = IIRSerializer::Format::Json;
  if(options_.SerializeIIR || (options_.DeserializeIIR != "")) {
    if(options_.IIRFormat == "json") {
      serializationKind = IIRSerializer::Format::Json;
    } else if(options_.IIRFormat == "byte") {
      serializationKind = IIRSerializer::Format::Byte;
    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }

  // Initialize optimizer
  OptimizerContext optimizer(getDiagnostics(), createOptimizerOptionsFromAllOptions(options_),
                             stencilInstantiationMap);

  // Optimization, step by step
  //===-----------------------------------------------------------------------------------------
  // broken but should run with no prerequesits
  optimizer.checkAndPushBack<PassSSA>();
  // rerun things we might have changed
  // optimizer.checkAndPushBack<PassFixVersionedInputFields>();
  // todo: this does not work since it does not check if it was already run
  //===-----------------------------------------------------------------------------------------
  // Plain diagnostics, should not even be a pass but is independent
  optimizer.checkAndPushBack<PassPrintStencilGraph>();
  //===-----------------------------------------------------------------------------------------
  // This is never used but if we want to reenable it, it is independent
  optimizer.checkAndPushBack<PassSetStageName>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassSetStageGraph>();
  optimizer.checkAndPushBack<PassSetDependencyGraph>();
  optimizer.checkAndPushBack<PassStageReordering>(reorderStrategy);
  // moved stages around ...
  optimizer.checkAndPushBack<PassSetSyncStage>();
  // if we want this info around, we should probably run this also
  // optimizer.checkAndPushBack<PassSetStageName>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassStageMerger>();
  // since this can change the scope of temporaries ...
  optimizer.checkAndPushBack<PassTemporaryType>();
  optimizer.checkAndPushBack<PassFixVersionedInputFields>();
  // modify stages and their extents ...
  optimizer.checkAndPushBack<PassComputeStageExtents>();
  // and changes their dependencies
  optimizer.checkAndPushBack<PassSetSyncStage>();
  //===-----------------------------------------------------------------------------------------
  // // should be irrelevant now
  // optimizer.checkAndPushBack<PassStencilSplitter>(maxFields);
  // // but would require a lot
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassTemporaryMerger>();
  // this should not affect the temporaries but since we're touching them it would probably be a
  // safe idea
  optimizer.checkAndPushBack<PassTemporaryType>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassInlining>(
      (getOptions().InlineSF || getOptions().PassTmpToFunction),
      PassInlining::InlineStrategy::ComputationsOnTheFly);
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassIntervalPartitioner>();
  // since this can change the scope of temporaries ...
  optimizer.checkAndPushBack<PassTemporaryType>();
  optimizer.checkAndPushBack<PassFixVersionedInputFields>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassTemporaryToStencilFunction>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassSetNonTempCaches>();
  // this should not affect the temporaries but since we're touching them it would probably be a
  // safe idea
  optimizer.checkAndPushBack<PassTemporaryType>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassSetCaches>();
  //===-----------------------------------------------------------------------------------------
  optimizer.checkAndPushBack<PassSetBlockSize>();
  //===-----------------------------------------------------------------------------------------
  // Plain diagnostics, should not even be a pass but is independent
  optimizer.checkAndPushBack<PassDataLocalityMetric>();
  //===-----------------------------------------------------------------------------------------

  DAWN_LOG(INFO) << "All the passes ran with the current command line arguments:";
  for(const auto& a : optimizer.getPassManager().getPasses()) {
    DAWN_LOG(INFO) << a->getName();
  }

  int i = 0;
  for(auto& stencil : optimizer.getStencilInstantiationMap()) {
    // Run optimization passes
    auto& instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting optimization and analysis passes for `" << instantiation->getName()
                   << "` ...";
    if(!optimizer.getPassManager().runAllPassesOnStencilInstantiation(optimizer, instantiation))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with optimization and analysis passes for `" << instantiation->getName()
                   << "`";

    if(options_.SerializeIIR) {
      const std::string originalFileName = remove_fileextension(
          options_.OutputFile.empty() ? instantiation->getMetaData().getFileName()
                                      : options_.OutputFile,
          ".cpp");
      IIRSerializer::serialize(originalFileName + "." + std::to_string(i) + ".iir", instantiation,
                               serializationKind);
      i++;
    }
    if(options_.DumpStencilInstantiation) {
      instantiation->dump();
    }
  }
  return optimizer.getStencilInstantiationMap();
}

std::unique_ptr<codegen::TranslationUnit>
DawnCompiler::generate(std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> const&
                           stencilInstantiationMap) {

  // Generate code
  if(options_.Backend == "gt" || options_.Backend == "gridtools") {
    codegen::gt::GTCodeGen CG(stencilInstantiationMap, diagnostics_, options_.UseParallelEP,
                              options_.MaxHaloPoints);
    return CG.generateCode();
  } else if(options_.Backend == "c++-naive") {
    codegen::cxxnaive::CXXNaiveCodeGen CG(stencilInstantiationMap, diagnostics_,
                                          options_.MaxHaloPoints);
    return CG.generateCode();
  } else if(options_.Backend == "c++-naive-ico") {
    codegen::cxxnaiveico::CXXNaiveIcoCodeGen CG(stencilInstantiationMap, diagnostics_,
                                                options_.MaxHaloPoints);
    return CG.generateCode();
  } else if(options_.Backend == "cuda") {
    const Array3i domain_size{options_.domain_size_i, options_.domain_size_j,
                              options_.domain_size_k};
    codegen::cuda::CudaCodeGen CG(stencilInstantiationMap, diagnostics_, options_.MaxHaloPoints,
                                  options_.nsms, options_.maxBlocksPerSM, domain_size);
    return CG.generateCode();
  } else if(options_.Backend == "c++-opt") {
    dawn_unreachable("GTClangOptCXX not supported yet");
  } else {
    diagnostics_.report(buildDiag("-backend", options_.Backend,
                                  "backend options must be : " +
                                      dawn::RangeToString(", ", "", "")(std::vector<std::string>{
                                          "gridtools", "c++-naive", "c++-opt", "c++-naive-ico"})));
    throw std::runtime_error("An error occurred.");
  }
}

std::unique_ptr<codegen::TranslationUnit>
DawnCompiler::compile(const std::shared_ptr<SIR>& stencilIR) {
  diagnostics_.clear();
  diagnostics_.setFilename(stencilIR->Filename);

  IIRSerializer::Format serializationKind = IIRSerializer::Format::Json;
  if(options_.SerializeIIR || (options_.DeserializeIIR != "")) {
    if(options_.IIRFormat == "json") {
      serializationKind = IIRSerializer::Format::Json;
    } else if(options_.IIRFormat == "byte") {
      serializationKind = IIRSerializer::Format::Byte;
    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }

  // Check if options are valid
  // -max-halo
  if(options_.MaxHaloPoints < 0) {
    diagnostics_.report(buildDiag("-max-halo", options_.MaxHaloPoints,
                                  "maximum number of allowed halo points must be >= 0"));
    throw std::runtime_error("An error occurred.");
  }

  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> stencilInstantiationMap;

  // TODO Make this clearer
  const bool inputIsSIR = options_.DeserializeIIR == "";
  if(inputIsSIR) {
    stencilInstantiationMap = parallelize(stencilIR);
  } else {
    // Initialize optimizer
    auto optimizerOptions = createOptimizerOptionsFromAllOptions(options_);

    OptimizerContext optimizer(getDiagnostics(), optimizerOptions, nullptr);
    auto instantiation = IIRSerializer::deserialize(options_.DeserializeIIR, serializationKind);
    optimizer.restoreIIR("<restored>", instantiation);

    stencilInstantiationMap = optimizer.getStencilInstantiationMap();
  }

  if(diagnostics_.hasErrors())
    throw std::runtime_error("An error occurred in the parallelizer.");

  if(!options_.Debug)
    stencilInstantiationMap = optimize(stencilInstantiationMap);

  if(diagnostics_.hasErrors())
    throw std::runtime_error("An error occurred in the optimizer.");

  return generate(stencilInstantiationMap);
}

const DiagnosticsEngine& DawnCompiler::getDiagnostics() const { return diagnostics_; }
DiagnosticsEngine& DawnCompiler::getDiagnostics() { return diagnostics_; }

const Options& DawnCompiler::getOptions() const { return options_; }
Options& DawnCompiler::getOptions() { return options_; }

} // namespace dawn
