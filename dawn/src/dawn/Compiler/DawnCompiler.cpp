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
#include "dawn/Optimizer/PassIntegrityCheck.h"
#include "dawn/Optimizer/PassIntervalPartitioner.h"
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
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/EditDistance.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"
#include "dawn/Support/StringUtil.h"
#include "dawn/Support/Unreachable.h"
#include "dawn/Validator/GridTypeChecker.h"
#include "dawn/Validator/LocationTypeChecker.h"

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

static OptimizerContext::OptimizerContextOptions
createOptimizerOptionsFromAllOptions(const Options& options) {
  OptimizerContext::OptimizerContextOptions retval;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  retval.NAME = options.NAME;
#include "dawn/Optimizer/OptimizerOptions.inc"
#undef OPT
  return retval;
}

DawnCompiler::DawnCompiler(Options* options) : diagnostics_(std::make_unique<DiagnosticsEngine>()) {
  options_ = options ? std::make_unique<Options>(*options) : std::make_unique<Options>();
}

std::unique_ptr<OptimizerContext> DawnCompiler::runOptimizer(std::shared_ptr<SIR> const& SIR) {
  // -reorder
  using ReorderStrategyKind = ReorderStrategy::Kind;
  ReorderStrategyKind reorderStrategy = StringSwitch<ReorderStrategyKind>(options_->ReorderStrategy)
                                            .Case("none", ReorderStrategyKind::None)
                                            .Case("greedy", ReorderStrategyKind::Greedy)
                                            .Case("scut", ReorderStrategyKind::Partitioning)
                                            .Default(ReorderStrategyKind::Unknown);

  if(reorderStrategy == ReorderStrategyKind::Unknown) {
    diagnostics_->report(
        buildDiag("-reorder", options_->ReorderStrategy, "", {"none", "greedy", "scut"}));
    return nullptr;
  }

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;
  MultistageSplitStrategy mssSplitStrategy;
  if(options_->MaxCutMSS) {
    mssSplitStrategy = MultistageSplitStrategy::MaxCut;
  } else {
    mssSplitStrategy = MultistageSplitStrategy::Optimized;
  }

  // -max-fields
  int maxFields = options_->MaxFieldsPerStencil;

  IIRSerializer::Format serializationKind = IIRSerializer::Format::Json;
  if(options_->SerializeIIR || (options_->DeserializeIIR != "")) {
    if(options_->IIRFormat == "json") {
      serializationKind = IIRSerializer::Format::Json;
    } else if(options_->IIRFormat == "byte") {
      serializationKind = IIRSerializer::Format::Byte;
    } else {
      dawn_unreachable("Unknown SIRFormat option");
    }
  }
  // Initialize optimizer
  OptimizerContext::OptimizerContextOptions optimizerOptions;
  if(options_) {
    optimizerOptions = createOptimizerOptionsFromAllOptions(*options_);
  }
  std::unique_ptr<OptimizerContext> optimizer;

  if(options_->DeserializeIIR == "") {
    optimizer = std::make_unique<OptimizerContext>(getDiagnostics(), optimizerOptions, SIR);
    optimizer->fillIIR();

    // Setup pass interface
    optimizer->checkAndPushBack<PassInlining>(true, PassInlining::InlineStrategy::InlineProcedures);
    // This pass is currently broken and needs to be redesigned before it can be enabled
    //  optimizer->checkAndPushBack<PassTemporaryFirstAccss>();
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
    // Run AST integrity checker pass after other optimizations
    optimizer->checkAndPushBack<PassIntegrityCheck>();

    DAWN_LOG(INFO) << "All the passes ran with the current command line arguments:";
    for(const auto& a : optimizer->getPassManager().getPasses()) {
      DAWN_LOG(INFO) << a->getName();
    }

    int i = 0;
    for(auto& stencil : optimizer->getStencilInstantiationMap()) {
      // Run optimization passes
      std::shared_ptr<iir::StencilInstantiation> instantiation = stencil.second;

      DAWN_LOG(INFO) << "Starting Optimization and Analysis passes for `"
                     << instantiation->getName() << "` ...";
      if(!optimizer->getPassManager().runAllPassesOnStencilInstantiation(*optimizer, instantiation))
        return nullptr;

      DAWN_LOG(INFO) << "Done with Optimization and Analysis passes for `"
                     << instantiation->getName() << "`";

      if(options_->SerializeIIR) {
        const fs::path p(options_->OutputFile.empty() ? instantiation->getMetaData().getFileName()
                                                      : options_->OutputFile);
        IIRSerializer::serialize(static_cast<std::string>(p.stem()) + "." + std::to_string(i) +
                                     ".iir",
                                 instantiation, serializationKind);
        i++;
      }
      if(options_->DumpStencilInstantiation) {
        instantiation->dump();
      }
    }
  } else {
    optimizer = std::make_unique<OptimizerContext>(getDiagnostics(), optimizerOptions, nullptr);
    auto instantiation =
        IIRSerializer::deserialize(options_->DeserializeIIR, optimizer.get(), serializationKind);
    optimizer->restoreIIR("<restored>", instantiation);
  }

  return optimizer;
} // namespace dawn

std::unique_ptr<codegen::TranslationUnit> DawnCompiler::compile(const std::shared_ptr<SIR>& SIR) {
  diagnostics_->clear();
  diagnostics_->setFilename(SIR->Filename);

  // Check if options are valid

  // -max-halo
  if(options_->MaxHaloPoints < 0) {
    diagnostics_->report(buildDiag("-max-halo", options_->MaxHaloPoints,
                                   "maximum number of allowed halo points must be >= 0"));
    return nullptr;
  }

  // SIR we received should be type consistent
  if(SIR->GridType == ast::GridType::Triangular) {
    LocationTypeChecker locationChecker;
    if(!locationChecker.checkLocationTypeConsistency(*SIR.get())) {
      DAWN_LOG(INFO) << "Location types in SIR are not consistent, no code generation";
      return nullptr;
    }
  }

  GridTypeChecker gridChecker;
  if(!gridChecker.checkGridTypeConsistency(*SIR.get())) {
    DAWN_LOG(INFO) << "Grid types in SIR are not consistent, no code generation";
    return nullptr;
  }

  // Initialize optimizer
  auto optimizer = runOptimizer(SIR);

  if(diagnostics_->hasErrors()) {
    DAWN_LOG(INFO) << "Errors occurred. Skipping code generation.";
    return nullptr;
  }

  // Generate code
  std::unique_ptr<codegen::CodeGen> CG;

  if(options_->Backend == "gt" || options_->Backend == "gridtools") {
    CG = std::make_unique<codegen::gt::GTCodeGen>(optimizer->getStencilInstantiationMap(),
                                                  *diagnostics_, options_->UseParallelEP,
                                                  options_->MaxHaloPoints);
  } else if(options_->Backend == "c++-naive") {
    CG = std::make_unique<codegen::cxxnaive::CXXNaiveCodeGen>(
        optimizer->getStencilInstantiationMap(), *diagnostics_, options_->MaxHaloPoints);
  } else if(options_->Backend == "c++-naive-ico") {
    CG = std::make_unique<codegen::cxxnaiveico::CXXNaiveIcoCodeGen>(
        optimizer->getStencilInstantiationMap(), *diagnostics_, options_->MaxHaloPoints);
  } else if(options_->Backend == "cuda") {
    const Array3i domain_size{options_->domain_size_i, options_->domain_size_j,
                              options_->domain_size_k};
    CG = std::make_unique<codegen::cuda::CudaCodeGen>(
        optimizer->getStencilInstantiationMap(), *diagnostics_, options_->MaxHaloPoints,
        options_->nsms, options_->maxBlocksPerSM, domain_size);
  } else if(options_->Backend == "c++-opt") {
    dawn_unreachable("GTClangOptCXX not supported yet");
  } else {
    diagnostics_->report(buildDiag("-backend", options_->Backend,
                                   "backend options must be : " +
                                       dawn::RangeToString(", ", "", "")(std::vector<std::string>{
                                           "gridtools", "c++-naive", "c++-opt", "c++-naive-ico"})));
    return nullptr;
  }

  return CG->generateCode();
}

const DiagnosticsEngine& DawnCompiler::getDiagnostics() const { return *diagnostics_.get(); }
DiagnosticsEngine& DawnCompiler::getDiagnostics() { return *diagnostics_.get(); }

const Options& DawnCompiler::getOptions() const { return *options_.get(); }
Options& DawnCompiler::getOptions() { return *options_.get(); }

} // namespace dawn
