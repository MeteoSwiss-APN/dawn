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

#include "dawn/Optimizer/Driver.h"
#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/TranslationUnit.h"
#include "dawn/Optimizer/Lowering.h"
#include "dawn/Optimizer/PassManager.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Exception.h"
#include "dawn/Support/Iterator.h"
#include "dawn/Support/Logger.h"
#include "dawn/Support/StringSwitch.h"

#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Optimizer/PassLocalVarType.h"
#include "dawn/Optimizer/PassManager.h"
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

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::shared_ptr<SIR>& stencilIR, const std::list<PassGroup>& groups,
    const Options& options) {

  auto stencilInstantiationMap = toStencilInstantiationMap(*stencilIR, options);

  PassManager passManager;

  using MultistageSplitStrategy = PassMultiStageSplitter::MultiStageSplittingStrategy;

  // required passes to have proper, parallelized IR
  passManager.pushBackPass<PassInlining>(PassInlining::InlineStrategy::InlineProcedures);
  passManager.pushBackPass<PassFieldVersioning>();
  passManager.pushBackPass<PassMultiStageSplitter>(
      options.MaxCutMSS ? MultistageSplitStrategy::MaxCut : MultistageSplitStrategy::Optimized);
  passManager.pushBackPass<PassTemporaryType>();
  passManager.pushBackPass<PassLocalVarType>();
  passManager.pushBackPass<PassRemoveScalars>();
  if(stencilIR->GridType == ast::GridType::Unstructured) {
    passManager.pushBackPass<PassStageSplitAllStatements>();
    passManager.pushBackPass<PassSetStageLocationType>();
  } else {
    passManager.pushBackPass<PassStageSplitter>();
  }
  passManager.pushBackPass<PassTemporaryType>();
  passManager.pushBackPass<PassFixVersionedInputFields>();
  passManager.pushBackPass<PassSetSyncStage>();
  // validation checks after parallelisation
  passManager.pushBackPass<PassValidation>();

  dawn::log::error.clear();
  for(auto& stencil : stencilInstantiationMap) {
    // Run optimization passes
    auto& instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting parallelization passes for `" << instantiation->getName()
                   << "` ...";
    if(!passManager.runAllPassesOnStencilInstantiation(instantiation, options))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with parallelization passes for `" << instantiation->getName() << "`";
  }

  if(dawn::log::error.size() > 0) {
    throw CompileError("An error occured in lowering");
  }

  return run(stencilInstantiationMap, groups, options);
}

std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const std::list<PassGroup>& groups, const Options& options) {

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

  PassManager passManager;

  for(auto group : groups) {
    switch(group) {
    case PassGroup::SSA:
      DAWN_ASSERT_MSG(false, "The SSA pass is broken.");
      // broken but should run with no prerequisites
      passManager.pushBackPass<PassSSA>();
      // rerun things we might have changed
      // passManager.pushBackPass<PassFixVersionedInputFields>();
      // todo: this does not work since it does not check if it was already run
      break;
    case PassGroup::PrintStencilGraph:
      passManager.pushBackPass<PassSetDependencyGraph>();
      // Plain diagnostics, should not even be a pass but is independent
      passManager.pushBackPass<PassPrintStencilGraph>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetStageName:
      // This is never used but if we want to reenable it, it is independent
      passManager.pushBackPass<PassSetStageName>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::StageReordering:
      if(stencilInstantiationMap.begin()->second->getIIR()->getGridType() !=
         ast::GridType::Unstructured) {
        passManager.pushBackPass<PassSetStageGraph>();
        passManager.pushBackPass<PassSetDependencyGraph>();
        passManager.pushBackPass<PassStageReordering>(reorderStrategy);
        // moved stages around ...
        passManager.pushBackPass<PassSetSyncStage>();
        // if we want this info around, we should probably run this also
        // passManager.pushBackPass<PassSetStageName>();
        // validation check
        passManager.pushBackPass<PassValidation>();
      } else {
        DAWN_LOG(WARNING) << "PassStageReordering currently disabled for unstructured meshes!";
      }
      break;
    case PassGroup::StageMerger:
      // merging requires the stage graph
      passManager.pushBackPass<PassSetStageGraph>();
      passManager.pushBackPass<PassSetDependencyGraph>();
      // running the actual pass
      passManager.pushBackPass<PassStageMerger>();
      // since this can change the scope of temporaries ...
      passManager.pushBackPass<PassTemporaryType>();
      passManager.pushBackPass<PassLocalVarType>();
      passManager.pushBackPass<PassRemoveScalars>();
      // modify stage dependencies
      passManager.pushBackPass<PassSetSyncStage>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::TemporaryMerger:
      passManager.pushBackPass<PassTemporaryMerger>();
      // this should not affect the temporaries but since we're touching them it would probably be
      // a safe idea
      passManager.pushBackPass<PassTemporaryType>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::Inlining:
      passManager.pushBackPass<PassInlining>(PassInlining::InlineStrategy::ComputationsOnTheFly);
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::IntervalPartitioning:
      passManager.pushBackPass<PassIntervalPartitioning>();
      // since this can change the scope of temporaries ...
      passManager.pushBackPass<PassTemporaryType>();
      // passManager.pushBackPass<PassFixVersionedInputFields>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::TmpToStencilFunction:
      passManager.pushBackPass<PassTemporaryToStencilFunction>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetNonTempCaches:
      passManager.pushBackPass<PassSetNonTempCaches>();
      // this should not affect the temporaries but since we're touching them it would probably be
      // a safe idea
      passManager.pushBackPass<PassTemporaryType>();
      passManager.pushBackPass<PassLocalVarType>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetCaches:
      passManager.pushBackPass<PassSetCaches>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::SetBlockSize:
      if(stencilInstantiationMap.begin()->second->getIIR()->getGridType() !=
         ast::GridType::Unstructured) {
        passManager.pushBackPass<PassSetBlockSize>();
        // validation check
        passManager.pushBackPass<PassValidation>();
      } else {
        DAWN_LOG(WARNING) << "PassSetBlockSize currently disabled for unstructured meshes!";
      }
      break;
    case PassGroup::DataLocalityMetric:
      // Plain diagnostics, should not even be a pass but is independent
      passManager.pushBackPass<PassDataLocalityMetric>();
      // validation check
      passManager.pushBackPass<PassValidation>();
      break;
    case PassGroup::Parallel:
      DAWN_ASSERT_MSG(false, "The parallel group is only valid for lowering to IIR.");
    }
  }
  // Note that we need to run PassInlining here if serializing or using the Cuda codegen backend.
  if(options.SerializeIIR) {
    passManager.pushBackPass<PassInlining>(PassInlining::InlineStrategy::ComputationsOnTheFly);
  }

  //===-----------------------------------------------------------------------------------------

  dawn::log::error.clear();
  for(auto& stencil : stencilInstantiationMap) {
    // Run optimization passes
    auto& instantiation = stencil.second;

    DAWN_LOG(INFO) << "Starting optimization and analysis passes for `" << instantiation->getName()
                   << "` ...";
    if(!passManager.runAllPassesOnStencilInstantiation(instantiation, options))
      throw std::runtime_error("An error occurred.");

    DAWN_LOG(INFO) << "Done with optimization and analysis passes for `" << instantiation->getName()
                   << "`";

    if(options.SerializeIIR) {
      const IIRSerializer::Format serializationKind =
          options.SerializeIIR ? IIRSerializer::parseFormatString(options.IIRFormat)
                               : IIRSerializer::Format::Json;
      IIRSerializer::serialize(instantiation->getName() + ".iir", instantiation, serializationKind);
    }

    if(options.DumpStencilInstantiation) {
      instantiation->dump(dawn::log::info.stream());
    }
  }

  if(dawn::log::error.size() > 0) {
    throw CompileError("An error occured in optimization");
  }

  return stencilInstantiationMap;
}

std::map<std::string, std::string> run(const std::string& sir, SIRSerializer::Format format,
                                       const std::list<PassGroup>& groups, const Options& options) {
  auto stencilIR = SIRSerializer::deserializeFromString(sir, format);
  auto optimizedSIM = run(stencilIR, groups, options);
  std::map<std::string, std::string> instantiationStringMap;
  for(auto [name, instantiation] : optimizedSIM) {
    instantiationStringMap.insert(std::make_pair(
        name, IIRSerializer::serializeToString(instantiation, IIRSerializer::Format::Json)));
  }
  return instantiationStringMap;
}

std::map<std::string, std::string>
run(const std::map<std::string, std::string>& stencilInstantiationMap, IIRSerializer::Format format,
    const std::list<PassGroup>& groups, const Options& options) {
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>> internalMap;
  for(auto [name, instStr] : stencilInstantiationMap) {
    internalMap.insert(std::make_pair(name, IIRSerializer::deserializeFromString(instStr, format)));
  }
  auto optimizedSIM = run(internalMap, groups, options);
  std::map<std::string, std::string> instantiationStringMap;
  for(auto [name, instantiation] : optimizedSIM) {
    instantiationStringMap.insert(std::make_pair(
        name, IIRSerializer::serializeToString(instantiation, IIRSerializer::Format::Json)));
  }
  return instantiationStringMap;
}

} // namespace dawn
