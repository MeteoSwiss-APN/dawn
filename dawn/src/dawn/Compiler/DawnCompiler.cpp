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
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/Optimizer/PassSetCaches.h"
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
#include "dawn/Optimizer/PassValidation.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/EditDistance.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"

namespace dawn {

namespace {

// CodeGen backends
enum class BackendType { GridTools, CXXNaive, CXXNaiveIco, CUDA, CXXOpt };

BackendType parseBackendString(const std::string& backendStr) {
  if(backendStr == "gt" || backendStr == "gridtools") {
    return BackendType::GridTools;
  } else if(backendStr == "naive" || backendStr == "cxxnaive" || backendStr == "c++-naive") {
    return BackendType::CXXNaive;
  } else if(backendStr == "ico" || backendStr == "naive-ico" || backendStr == "c++-naive-ico") {
    return BackendType::CXXNaiveIco;
  } else if(backendStr == "cuda" || backendStr == "CUDA") {
    return BackendType::CUDA;
  } else {
    throw CompileError("Backend not supported");
  }
}

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

OptimizerContext::OptimizerContextOptions
createOptimizerOptionsFromAllOptions(const Options& options) {
  OptimizerContext::OptimizerContextOptions retval;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  retval.NAME = options.NAME;
#include "dawn/Optimizer/OptimizerOptions.inc"
#undef OPT
  return retval;
}

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

} // namespace

DawnCompiler::DawnCompiler(const Options& options) : diagnostics_(), options_(options) {}

std::unique_ptr<OptimizerContext> DawnCompiler::runOptimizer(std::shared_ptr<SIR> const& SIR) {
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
    return nullptr;
  }

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;
  MultistageSplitStrategy mssSplitStrategy;
  if(options_.MaxCutMSS) {
    mssSplitStrategy = MultistageSplitStrategy::MaxCut;
  } else {
    mssSplitStrategy = MultistageSplitStrategy::Optimized;
  }

  // -max-fields
  int maxFields = options_.MaxFieldsPerStencil;

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
  auto optimizerOptions = createOptimizerOptionsFromAllOptions(options_);
  std::unique_ptr<OptimizerContext> optimizer;

  PassValidation validationPass(*optimizer);
  validationPass.run(SIR);

  if(options_.DeserializeIIR == "") {
    optimizer = std::make_unique<OptimizerContext>(getDiagnostics(), optimizerOptions, SIR);

    // Setup pass interface
    optimizer->checkAndPushBack<PassInlining>(true, PassInlining::InlineStrategy::InlineProcedures);
    // This pass is currently broken and needs to be redesigned before it can be enabled
    //  optimizer->checkAndPushBack<PassTemporaryFirstAccss>();
    optimizer->checkAndPushBack<PassFieldVersioning>();
    optimizer->checkAndPushBack<PassSSA>();
    optimizer->checkAndPushBack<PassLocalVarType>(); // Needs to be run before splitters.
    optimizer->checkAndPushBack<PassMultiStageSplitter>(mssSplitStrategy);
    optimizer->checkAndPushBack<PassStageSplitter>();
    optimizer->checkAndPushBack<PassPrintStencilGraph>();
    optimizer->checkAndPushBack<PassTemporaryType>();
    optimizer->checkAndPushBack<PassLocalVarType>(); // Needs to be run after temporary type.
    optimizer->checkAndPushBack<PassSetStageName>();
    optimizer->checkAndPushBack<PassSetStageGraph>();
    optimizer->checkAndPushBack<PassStageReordering>(reorderStrategy);
    optimizer->checkAndPushBack<PassStageMerger>();
    optimizer->checkAndPushBack<PassStencilSplitter>(maxFields);
    optimizer->checkAndPushBack<PassTemporaryType>();
    optimizer->checkAndPushBack<PassLocalVarType>(); // Needs to be run after temporary type.
    optimizer->checkAndPushBack<PassTemporaryMerger>();
    optimizer->checkAndPushBack<PassInlining>(
        (getOptions().InlineSF || getOptions().PassTmpToFunction),
        PassInlining::InlineStrategy::ComputationsOnTheFly);
    optimizer->checkAndPushBack<PassIntervalPartitioner>();
    optimizer->checkAndPushBack<PassTemporaryToStencilFunction>();
    optimizer->checkAndPushBack<PassSetNonTempCaches>();
    optimizer->checkAndPushBack<PassSetCaches>();
    optimizer->checkAndPushBack<PassFixVersionedInputFields>();
    optimizer->checkAndPushBack<PassComputeStageExtents>();
    // This pass is disabled because the boundary conditions need to be fixed.
    // optimizer->checkAndPushBack<PassSetBoundaryCondition>();
    if(getOptions().Backend == "cuda") {
      optimizer->checkAndPushBack<PassSetBlockSize>();
    }
    optimizer->checkAndPushBack<PassDataLocalityMetric>();
    optimizer->checkAndPushBack<PassSetSyncStage>();
    // Since both cuda code generation as well as serialization do not support stencil-functions, we
    // need to inline here as the last step
    optimizer->checkAndPushBack<PassInlining>(getOptions().Backend == "cuda" ||
                                                  getOptions().SerializeIIR,
                                              PassInlining::InlineStrategy::ComputationsOnTheFly);

    DAWN_LOG(INFO) << "All the passes ran with the current command line arguments:";
    for(const auto& a : optimizer->getPassManager().getPasses()) {
      DAWN_LOG(INFO) << a->getName();
    }

    int i = 0;
    for(auto& [_name, instantiation] : optimizer->getStencilInstantiationMap()) {
      DAWN_LOG(INFO) << "Starting Optimization and Analysis passes for `"
                     << instantiation->getName() << "` ...";
      if(!optimizer->getPassManager().runAllPassesOnStencilInstantiation(*optimizer, instantiation))
        return nullptr;

      DAWN_LOG(INFO) << "Done with Optimization and Analysis passes for `"
                     << instantiation->getName() << "`";

      if(options_.SerializeIIR) {
        const fs::path p(options_.OutputFile.empty() ? instantiation->getMetaData().getFileName()
                                                     : options_.OutputFile);
        IIRSerializer::serialize(static_cast<std::string>(p.stem()) + "." + std::to_string(i) +
                                     ".iir",
                                 instantiation, serializationKind);
        i++;
      }
      if(options_.DumpStencilInstantiation) {
        instantiation->dump();
      }
    }
  } else {
    optimizer = std::make_unique<OptimizerContext>(getDiagnostics(), optimizerOptions, nullptr);
    auto instantiation =
        IIRSerializer::deserialize(options_.DeserializeIIR, optimizer.get(), serializationKind);
    optimizer->restoreIIR("<restored>", instantiation);
  }

  return optimizer;
}

std::unique_ptr<codegen::TranslationUnit>
DawnCompiler::generate(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
                           stencilInstantiationMap) {
  // Generate code
  try {
    BackendType backend = parseBackendString(options_.Backend);
    switch(backend) {
    case BackendType::GridTools: {
      codegen::gt::GTCodeGen CG(stencilInstantiationMap, diagnostics_, options_.UseParallelEP,
                                options_.MaxHaloPoints);
      return CG.generateCode();
    }
    case BackendType::CXXNaive: {
      codegen::cxxnaive::CXXNaiveCodeGen CG(stencilInstantiationMap, diagnostics_,
                                            options_.MaxHaloPoints);
      return CG.generateCode();
    }
    case BackendType::CUDA: {
      const Array3i domain_size{options_.domain_size_i, options_.domain_size_j,
                                options_.domain_size_k};
      codegen::cuda::CudaCodeGen CG(stencilInstantiationMap, diagnostics_, options_.MaxHaloPoints,
                                    options_.nsms, options_.maxBlocksPerSM, domain_size);
      return CG.generateCode();
    }
    case BackendType::CXXNaiveIco: {
      codegen::cxxnaiveico::CXXNaiveIcoCodeGen CG(stencilInstantiationMap, diagnostics_,
                                                  options_.MaxHaloPoints);

      return CG.generateCode();
    }
    case BackendType::CXXOpt:
      dawn_unreachable("GTClangOptCXX not supported yet");
    }
  } catch(...) {
    diagnostics_.report(buildDiag("-backend", options_.Backend,
                                  "backend options must be : " +
                                      dawn::RangeToString(", ", "", "")(std::vector<std::string>{
                                          "gridtools", "c++-naive", "c++-opt", "c++-naive-ico"})));
    return nullptr;
  }
  return nullptr;
}

std::unique_ptr<codegen::TranslationUnit> DawnCompiler::compile(const std::shared_ptr<SIR>& SIR) {
  diagnostics_.clear();
  diagnostics_.setFilename(SIR->Filename);

  // Check if options are valid

  // -max-halo
  if(options_.MaxHaloPoints < 0) {
    diagnostics_.report(buildDiag("-max-halo", options_.MaxHaloPoints,
                                  "maximum number of allowed halo points must be >= 0"));
    return nullptr;
  }

  // Initialize optimizer
  auto optimizer = runOptimizer(SIR);

  if(diagnostics_.hasErrors()) {
    DAWN_LOG(INFO) << "Errors occurred. Skipping code generation.";
    return nullptr;
  } else {
    DAWN_ASSERT_MSG(optimizer, "No errors, but optimizer context fails to exist!");
  }

  return generate(optimizer->getStencilInstantiationMap());
}

const DiagnosticsEngine& DawnCompiler::getDiagnostics() const { return diagnostics_; }
DiagnosticsEngine& DawnCompiler::getDiagnostics() { return diagnostics_; }

const Options& DawnCompiler::getOptions() const { return options_; }
Options& DawnCompiler::getOptions() { return options_; }

} // namespace dawn
