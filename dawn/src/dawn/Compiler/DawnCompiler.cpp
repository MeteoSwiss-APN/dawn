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
#include "dawn/AST/GridType.h"
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
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
#include "dawn/Optimizer/PassSetLoopOrder.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageLocationType.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitAllStatements.h"
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
#include "dawn/Support/Exception.h"
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
#include "dawn/CodeGen/Options.inc"
#include "dawn/Optimizer/Options.inc"
#include "dawn/Optimizer/PassOptions.inc"
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

std::list<PassGroup> DawnCompiler::defaultPassGroups() {
  return {PassGroup::SetStageName, PassGroup::StageReordering, PassGroup::StageMerger,
          PassGroup::SetCaches, PassGroup::SetBlockSize};
}

DawnCompiler::DawnCompiler(const Options& options) : diagnostics_(), options_(options) {}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
DawnCompiler::lowerToIIR(const std::shared_ptr<SIR>& stencilIR) {
  diagnostics_.clear();
  diagnostics_.setFilename(stencilIR->Filename);

  OptimizerContext optimizer(getDiagnostics(), createOptimizerOptionsFromAllOptions(options_),
                             stencilIR);

  // required passes to have proper, parallelized IR
  optimizer.pushBackPass<PassInlining>(PassInlining::InlineStrategy::InlineProcedures);
  optimizer.pushBackPass<PassFieldVersioning>();
  optimizer.pushBackPass<PassTemporaryType>();
  optimizer.pushBackPass<PassLocalVarType>();
  optimizer.pushBackPass<PassRemoveScalars>();
  optimizer.pushBackPass<PassStageSplitAllStatements>();
  if(stencilIR->GridType == ast::GridType::Unstructured) {
    optimizer.pushBackPass<PassSetStageLocationType>();
  }
  optimizer.pushBackPass<PassTemporaryType>();
  optimizer.pushBackPass<PassFixVersionedInputFields>();
  optimizer.pushBackPass<PassComputeStageExtents>();
  optimizer.pushBackPass<PassSetSyncStage>();
  // validation checks after parallelisation
  optimizer.pushBackPass<PassValidation>();

  for(auto& stencil : optimizer.getStencilInstantiationMap()) {
    // Run optimization passes
    auto& instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting parallelization passes for `" << instantiation->getName()
                   << "` ...";
    if(!optimizer.getPassManager().runAllPassesOnStencilInstantiation(optimizer, instantiation))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with parallelization passes for `" << instantiation->getName() << "`";
  }

  return optimizer.getStencilInstantiationMap();
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
DawnCompiler::optimize(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
                           stencilInstantiationMap,
                       const std::list<PassGroup>& groups) {
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

  for(auto group : groups) {
    switch(group) {
    case PassGroup::SSA:
      DAWN_ASSERT_MSG(false, "The SSA pass is broken.");
      // broken but should run with no prerequisites
      optimizer.pushBackPass<PassSSA>();
      // rerun things we might have changed
      // optimizer.pushBackPass<PassFixVersionedInputFields>();
      // todo: this does not work since it does not check if it was already run
      break;
    case PassGroup::PrintStencilGraph:
      optimizer.pushBackPass<PassSetDependencyGraph>();
      // Plain diagnostics, should not even be a pass but is independent
      optimizer.pushBackPass<PassPrintStencilGraph>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetStageName:
      // This is never used but if we want to reenable it, it is independent
      optimizer.pushBackPass<PassSetStageName>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::StageReordering:
      optimizer.pushBackPass<PassSetStageGraph>();
      optimizer.pushBackPass<PassSetDependencyGraph>();
      optimizer.pushBackPass<PassStageReordering>(reorderStrategy);
      // moved stages around ...
      optimizer.pushBackPass<PassSetSyncStage>();
      // if we want this info around, we should probably run this also
      // optimizer.pushBackPass<PassSetStageName>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::StageMerger:
      // merging requires the stage graph
      optimizer.pushBackPass<PassSetStageGraph>();
      optimizer.pushBackPass<PassSetDependencyGraph>();
      // running the actual pass
      optimizer.pushBackPass<PassStageMerger>();
      // since this can change the scope of temporaries ...
      optimizer.pushBackPass<PassTemporaryType>();
      optimizer.pushBackPass<PassLocalVarType>();
      optimizer.pushBackPass<PassRemoveScalars>();
      // modify stages and their extents ...
      optimizer.pushBackPass<PassComputeStageExtents>();
      // and changes their dependencies
      optimizer.pushBackPass<PassSetSyncStage>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::TemporaryMerger:
      optimizer.pushBackPass<PassTemporaryMerger>();
      // this should not affect the temporaries but since we're touching them it would probably be a
      // safe idea
      optimizer.pushBackPass<PassTemporaryType>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::Inlining:
      optimizer.pushBackPass<PassInlining>(PassInlining::InlineStrategy::ComputationsOnTheFly);
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::IntervalPartitioning:
      optimizer.pushBackPass<PassIntervalPartitioning>();
      // since this can change the scope of temporaries ...
      optimizer.pushBackPass<PassTemporaryType>();
      // optimizer.pushBackPass<PassFixVersionedInputFields>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::TmpToStencilFunction:
      optimizer.pushBackPass<PassTemporaryToStencilFunction>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetNonTempCaches:
      optimizer.pushBackPass<PassSetNonTempCaches>();
      // this should not affect the temporaries but since we're touching them it would probably be a
      // safe idea
      optimizer.pushBackPass<PassTemporaryType>();
      optimizer.pushBackPass<PassLocalVarType>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetCaches:
      optimizer.pushBackPass<PassSetCaches>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetBlockSize:
      optimizer.pushBackPass<PassSetBlockSize>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::DataLocalityMetric:
      // Plain diagnostics, should not even be a pass but is independent
      optimizer.pushBackPass<PassDataLocalityMetric>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetLoopOrder:
      // Plain diagnostics, should not even be a pass but is independent
      optimizer.pushBackPass<PassSetLoopOrder>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::Parallel:
      DAWN_ASSERT_MSG(false, "The parallel group is only valid for lowering to IIR.");
    }
  }
  if(options_.Backend == "cuda" || options_.SerializeIIR) {
    optimizer.pushBackPass<PassInlining>(PassInlining::InlineStrategy::ComputationsOnTheFly);
    // validation check
    optimizer.pushBackPass<PassValidation>();
  }

  //===-----------------------------------------------------------------------------------------

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
      const auto p =
          fs::path(options_.OutputFile.empty() ? instantiation->getMetaData().getFileName()
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

  return optimizer.getStencilInstantiationMap();
}

std::unique_ptr<codegen::TranslationUnit>
DawnCompiler::generate(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
                           stencilInstantiationMap) {
  // Generate code
  BackendType backend;
  try {
    backend = parseBackendString(options_.Backend);
  } catch(CompileError& e) {
    diagnostics_.report(buildDiag("-backend", options_.Backend,
                                  "backend options must be : " +
                                      dawn::RangeToString(", ", "", "")(std::vector<std::string>{
                                          "gridtools", "c++-naive", "c++-opt", "c++-naive-ico"})));
    return nullptr;
  }
  try {
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
      const Array3i domain_size{options_.DomainSizeI, options_.DomainSizeJ, options_.DomainSizeK};
      codegen::cuda::CudaCodeGen CG(stencilInstantiationMap, diagnostics_, options_.MaxHaloPoints,
                                    options_.nsms, options_.MaxBlocksPerSM, domain_size);
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
    DiagnosticsBuilder diag(DiagnosticsKind::Error);
    diag << "code generation for backend `" << options_.Backend << "` failed";
    diagnostics_.report(diag);
  }
  return nullptr;
}

std::unique_ptr<codegen::TranslationUnit>
DawnCompiler::compile(const std::shared_ptr<SIR>& stencilIR, std::list<PassGroup> groups) {
  if(groups.empty())
    groups = defaultPassGroups();

  // Parallelize the SIR
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> SIM, optimizedSIM;
  try {
    SIM = lowerToIIR(stencilIR);
  } catch(...) {
    DAWN_LOG(INFO) << "Errors occurred. Skipping optimisation and code generation.";
    return nullptr;
  }

  // Optimize the IIR
  try {
    optimizedSIM = optimize(SIM, groups);
  } catch(...) {
    DAWN_LOG(INFO) << "Errors occurred. Skipping code generation.";
    return nullptr;
  }

  // Generate the Code
  return generate(optimizedSIM);
}

const DiagnosticsEngine& DawnCompiler::getDiagnostics() const { return diagnostics_; }
DiagnosticsEngine& DawnCompiler::getDiagnostics() { return diagnostics_; }

const Options& DawnCompiler::getOptions() const { return options_; }
Options& DawnCompiler::getOptions() { return options_; }

} // namespace dawn
