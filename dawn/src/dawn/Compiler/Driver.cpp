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

#include "dawn/Compiler/Driver.h"
#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Iterator.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringSwitch.h"

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
#include "dawn/Optimizer/PassRemoveScalars.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/Optimizer/PassSetBoundaryCondition.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetDependencyGraph.h"
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
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

namespace dawn {

std::list<PassGroup> defaultPassGroups() {
  return {PassGroup::SetStageName, PassGroup::StageReordering, PassGroup::StageMerger,
          PassGroup::SetCaches, PassGroup::SetBlockSize};
}

std::list<std::string> defaultPassGroupsStrings() {
  return {"SetStageName", "StageReordering", "StageMerger", "SetCaches", "SetBlockSize"};
}

PassGroup parsePassGroupString(const std::string& passGroup) {
  if(passGroup == "SSA" || passGroup == "ssa")
    return dawn::PassGroup::SSA;
  else if(passGroup == "PrintStencilGraph" || passGroup == "print-stencil-graph")
    return dawn::PassGroup::PrintStencilGraph;
  else if(passGroup == "SetStageName" || passGroup == "set-stage-name")
    return dawn::PassGroup::SetStageName;
  else if(passGroup == "StageReordering" || passGroup == "stage-reordering")
    return dawn::PassGroup::StageReordering;
  else if(passGroup == "StageMerger" || passGroup == "stage-merger")
    return dawn::PassGroup::StageMerger;
  else if(passGroup == "TemporaryMerger" || passGroup == "temporary-merger" ||
          passGroup == "tmp-merger")
    return dawn::PassGroup::TemporaryMerger;
  else if(passGroup == "Inlining" || passGroup == "inlining")
    return dawn::PassGroup::Inlining;
  else if(passGroup == "IntervalPartitioning" || passGroup == "interval-partitioning")
    return dawn::PassGroup::IntervalPartitioning;
  else if(passGroup == "TmpToStencilFunction" || passGroup == "tmp-to-stencil-function" ||
          passGroup == "tmp-to-stencil-fcn" || passGroup == "tmp-to-function" ||
          passGroup == "tmp-to-fcn")
    return dawn::PassGroup::TmpToStencilFunction;
  else if(passGroup == "SetNonTempCaches" || passGroup == "set-non-tmp-caches" ||
          passGroup == "set-nontmp-caches")
    return dawn::PassGroup::SetNonTempCaches;
  else if(passGroup == "SetCaches" || passGroup == "set-caches")
    return dawn::PassGroup::SetCaches;
  else if(passGroup == "SetBlockSize" || passGroup == "set-block-size")
    return dawn::PassGroup::SetBlockSize;
  else if(passGroup == "DataLocalityMetric" || passGroup == "data-locality-metric")
    return dawn::PassGroup::DataLocalityMetric;
  else
    throw std::invalid_argument(std::string("Unknown pass group: ") + passGroup);
}

namespace {

OptimizerContext::OptimizerContextOptions createOptionsFromOptions(const Options& options) {
  OptimizerContext::OptimizerContextOptions retval;
#define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP) \
  retval.NAME = options.NAME;
#include "dawn/Optimizer/Options.inc"
#undef OPT
  return retval;
}

} // namespace

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups,
    const Options& options) {
  DiagnosticsEngine diag;
  diag.setFilename(stencilIR->Filename);

  OptimizerContext optimizer(diag, createOptionsFromOptions(options), stencilIR);

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;

  // required passes to have proper, parallelized IR
  optimizer.pushBackPass<PassInlining>(PassInlining::InlineStrategy::InlineProcedures);
  optimizer.pushBackPass<PassFieldVersioning>();
  optimizer.pushBackPass<PassMultiStageSplitter>(
      options.MaxCutMSS ? MultistageSplitStrategy::MaxCut : MultistageSplitStrategy::Optimized);
  optimizer.pushBackPass<PassTemporaryType>();
  optimizer.pushBackPass<PassLocalVarType>();
  optimizer.pushBackPass<PassRemoveScalars>();
  if(stencilIR->GridType == ast::GridType::Unstructured) {
    optimizer.pushBackPass<PassStageSplitAllStatements>();
    optimizer.pushBackPass<PassSetStageLocationType>();
  } else {
    optimizer.pushBackPass<PassStageSplitter>();
  }
  optimizer.pushBackPass<PassTemporaryType>();
  optimizer.pushBackPass<PassFixVersionedInputFields>();
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

  return run(optimizer.getStencilInstantiationMap(), groups, options);
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const std::list<PassGroup>& groups, const Options& options) {
  DiagnosticsEngine diag;

  // -reorder
  using ReorderStrategyKind = ReorderStrategy::Kind;
  ReorderStrategyKind reorderStrategy = StringSwitch<ReorderStrategyKind>(options.ReorderStrategy)
                                            .Case("none", ReorderStrategyKind::None)
                                            .Case("greedy", ReorderStrategyKind::Greedy)
                                            .Case("scut", ReorderStrategyKind::Partitioning)
                                            .Default(ReorderStrategyKind::Unknown);

  if(reorderStrategy == ReorderStrategyKind::Unknown) {
    throw std::invalid_argument(std::string("Unknown ReorderStrategy") + options.ReorderStrategy +
                                ". Options are {none, greedy, scut}.");
  }

  // Initialize optimizer
  OptimizerContext optimizer(diag, createOptionsFromOptions(options), stencilInstantiationMap);

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
      // modify stage dependencies
      optimizer.pushBackPass<PassSetSyncStage>();
      // validation check
      optimizer.pushBackPass<PassValidation>();
      break;
    case PassGroup::TemporaryMerger:
      optimizer.pushBackPass<PassTemporaryMerger>();
      // this should not affect the temporaries but since we're touching them it would probably be
      // a safe idea
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
      // this should not affect the temporaries but since we're touching them it would probably be
      // a safe idea
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
    case PassGroup::Parallel:
      DAWN_ASSERT_MSG(false, "The parallel group is only valid for lowering to IIR.");
    }
  }
  // Note that we need to run PassInlining here if serializing or using the Cuda codegen backend.
  if(options.SerializeIIR) {
    optimizer.pushBackPass<PassInlining>(PassInlining::InlineStrategy::ComputationsOnTheFly);
  }

  //===-----------------------------------------------------------------------------------------

  for(auto& stencil : optimizer.getStencilInstantiationMap()) {
    // Run optimization passes
    auto& instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting optimization and analysis passes for `" << instantiation->getName()
                   << "` ...";
    if(!optimizer.getPassManager().runAllPassesOnStencilInstantiation(optimizer, instantiation))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with optimization and analysis passes for `" << instantiation->getName()
                   << "`";

    if(options.SerializeIIR) {
      const dawn::IIRSerializer::Format serializationKind =
          options.SerializeIIR ? dawn::IIRSerializer::parseFormatString(options.IIRFormat)
                               : dawn::IIRSerializer::Format::Json;
      dawn::IIRSerializer::serialize(instantiation->getName() + ".iir", instantiation,
                                     serializationKind);
    }

    if(options.DumpStencilInstantiation) {
      instantiation->dump();
    }
  }

  return optimizer.getStencilInstantiationMap();
}

std::map<std::string, std::string> run(const std::string& sir, const std::string& format,
                                       const std::list<std::string>& groups,
                                       const Options& options) {
  auto stencilIR =
      SIRSerializer::deserializeFromString(sir, SIRSerializer::parseFormatString(format));
  std::list<PassGroup> passGroup;
  std::transform(std::begin(groups), std::end(groups),
                 std::inserter(passGroup, std::end(passGroup)),
                 [](const std::string& group) { return parsePassGroupString(group); });
  auto optimizedSIM = run(stencilIR, passGroup, options);
  std::map<std::string, std::string> instantiationStringMap;
  const IIRSerializer::Format outputFormat = IIRSerializer::Format::Json;
  for(auto [name, instantiation] : optimizedSIM) {
    instantiationStringMap.insert(
        std::make_pair(name, dawn::IIRSerializer::serializeToString(instantiation, outputFormat)));
  }
  return instantiationStringMap;
}

std::map<std::string, std::string>
run(const std::map<std::string, std::string>& stencilInstantiationMap, const std::string& format,
    const std::list<std::string>& groups, const dawn::Options& options) {
  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> internalMap;
  for(auto [name, instStr] : stencilInstantiationMap) {
    internalMap.insert(std::make_pair(
        name,
        IIRSerializer::deserializeFromString(instStr, IIRSerializer::parseFormatString(format))));
  }
  std::list<PassGroup> passGroup;
  std::transform(std::begin(groups), std::end(groups),
                 std::inserter(passGroup, std::end(passGroup)),
                 [](const std::string& group) { return parsePassGroupString(group); });
  auto optimizedSIM = dawn::run(internalMap, passGroup, options);
  std::map<std::string, std::string> instantiationStringMap;
  for(auto [name, instantiation] : optimizedSIM) {
    instantiationStringMap.insert(std::make_pair(
        name,
        dawn::IIRSerializer::serializeToString(instantiation, dawn::IIRSerializer::Format::Json)));
  }
  return instantiationStringMap;
}

std::unique_ptr<codegen::TranslationUnit> compile(const std::shared_ptr<SIR>& stencilIR,
                                                  const std::list<PassGroup>& passGroups,
                                                  const Options& optimizerOptions,
                                                  codegen::Backend backend,
                                                  const codegen::Options& codegenOptions) {
  return codegen::run(run(stencilIR, passGroups, optimizerOptions), backend, codegenOptions);
}

std::string compile(const std::string& sir, const std::string& format,
                    const std::list<std::string>& groups, const Options& optimizerOptions,
                    const std::string& backend, const codegen::Options& codegenOptions) {
  // Could call string version here, but that forces serialization of the IIR. Avoids serializing.
  auto stencilIR =
      SIRSerializer::deserializeFromString(sir, SIRSerializer::parseFormatString(format));
  std::list<PassGroup> passGroup;
  std::transform(std::begin(groups), std::end(groups),
                 std::inserter(passGroup, std::end(passGroup)),
                 [](const std::string& group) { return parsePassGroupString(group); });
  auto optimizedSIM = run(stencilIR, passGroup, optimizerOptions);
  return codegen::generate(
      codegen::run(optimizedSIM, codegen::parseBackendString(backend), codegenOptions));
}

} // namespace dawn
