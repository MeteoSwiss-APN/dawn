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

#include "dawn/Unittest/CompilerUtil.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassDataLocalityMetric.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassIntervalPartitioner.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetBlockSize.h"
#include "dawn/Optimizer/PassSetCaches.h"
#include "dawn/Optimizer/PassSetNonTempCaches.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/PassTemporaryType.h"

namespace dawn {

bool CompilerUtil::Verbose;

dawn::DiagnosticsEngine CompilerUtil::diag_;

std::shared_ptr<SIR> CompilerUtil::load(const std::string& sirFilename) {
  return SIRSerializer::deserialize(sirFilename);
}

std::shared_ptr<iir::StencilInstantiation>
CompilerUtil::load(const std::string& irFilename,
                   const dawn::OptimizerContext::OptimizerContextOptions& options,
                   std::unique_ptr<OptimizerContext>& context, const std::string& envPath) {
  std::string filename = envPath;
  if(!filename.empty())
    filename += "/";
  filename += irFilename;

  if(filename.find(".sir") != std::string::npos) {
    return lower(filename, options, context, envPath);
  } else {
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    context = std::make_unique<OptimizerContext>(diag_, options, sir);
    return IIRSerializer::deserialize(filename);
  }
}

std::shared_ptr<iir::StencilInstantiation>
CompilerUtil::lower(const std::shared_ptr<dawn::SIR>& sir,
                    const dawn::OptimizerContext::OptimizerContextOptions& options,
                    std::unique_ptr<OptimizerContext>& context) {
  //dawn::OptimizerContext::OptimizerContextOptions options;
  context = std::make_unique<OptimizerContext>(diag_, options, sir);
  std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& map =
      context->getStencilInstantiationMap();
  return map.begin()->second;
}

std::shared_ptr<iir::StencilInstantiation>
CompilerUtil::lower(const std::string& sirFilename,
                    const dawn::OptimizerContext::OptimizerContextOptions& options,
                    std::unique_ptr<OptimizerContext>& context,
                    const std::string& envPath) {
  std::string filename = envPath;
  if(!filename.empty())
    filename += "/";
  filename += sirFilename;

  // std::shared_ptr<dawn::SIR> sir = nullptr;
  std::shared_ptr<dawn::SIR> sir = load(filename);
  return lower(sir, options, context);
}

stencilInstantiationContext CompilerUtil::compile(const std::shared_ptr<SIR>& sir) {
  dawn::Options options;
  DawnCompiler compiler(options);
  auto optimizer = compiler.runOptimizer(sir);

  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue()) {
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    }
    throw std::runtime_error("Compilation failed");
  }

  return optimizer->getStencilInstantiationMap();
}

stencilInstantiationContext CompilerUtil::compile(const std::string& sirFile) {
  std::shared_ptr<SIR> sir = load(sirFile);
  return compile(sir);
}

void CompilerUtil::clearDiags() {
  diag_.clear();
}

void CompilerUtil::dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  using CG = dawn::codegen::cxxnaive::CXXNaiveCodeGen;
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0);
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

void CompilerUtil::dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx) {
  using CG = dawn::codegen::cuda::CudaCodeGen;
  dawn::DiagnosticsEngine diagnostics;
  CG generator(ctx, diagnostics, 0, 0, 0, {0, 0, 0});
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

std::vector<std::shared_ptr<Pass>>
CompilerUtil::createGroup(PassGroup group, std::unique_ptr<OptimizerContext>& context) {
  auto mssSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy::Optimized;
  auto inlineStrategy = dawn::PassInlining::InlineStrategy::InlineProcedures;
  auto reorderStrategy = dawn::ReorderStrategy::Kind::Greedy;
  auto inlineOnTheFly = dawn::PassInlining::InlineStrategy::ComputationsOnTheFly;

  std::vector<std::shared_ptr<Pass>> passes;
  switch(group) {
  case PassGroup::Parallel:
    addPass<dawn::PassInlining>(context, passes, true, inlineStrategy);
    addPass<dawn::PassFieldVersioning>(context, passes);
    addPass<dawn::PassSSA>(context, passes); // Did I do this for a reason?
    addPass<dawn::PassMultiStageSplitter>(context, passes, mssSplitStrategy);
    addPass<dawn::PassStageSplitter>(context, passes);
    addPass<dawn::PassTemporaryType>(context, passes);
    addPass<dawn::PassFixVersionedInputFields>(context, passes);
    addPass<dawn::PassComputeStageExtents>(context, passes);
    addPass<dawn::PassSetSyncStage>(context, passes);
    break;

  case PassGroup::ReorderStages:
    addPass<dawn::PassSetStageGraph>(context, passes);
    addPass<dawn::PassStageReordering>(context, passes, reorderStrategy);
    addPass<dawn::PassSetSyncStage>(context, passes);
    addPass<dawn::PassSetStageName>(context, passes);
    break;

  case PassGroup::MergeStages:
    addPass<dawn::PassStageMerger>(context, passes);
    // since this can change the scope of temporaries ...
    addPass<dawn::PassTemporaryType>(context, passes);
    addPass<dawn::PassFixVersionedInputFields>(context, passes);
    // modify stages and their extents ...
    addPass<dawn::PassComputeStageExtents>(context, passes);
    // and changes their dependencies
    addPass<dawn::PassSetSyncStage>(context, passes);
    break;

  case PassGroup::MergeTemporaries:
    addPass<dawn::PassTemporaryMerger>(context, passes);
    // this should not affect the temporaries but since we're touching them it would probably be a
    // safe idea
    addPass<dawn::PassTemporaryType>(context, passes);
    break;

  case PassGroup::Inlining:
    addPass<dawn::PassInlining>(context, passes, false, inlineOnTheFly);
    break;

  case PassGroup::PartitionIntervals:
    addPass<dawn::PassIntervalPartitioner>(context, passes);
    // since this can change the scope of temporaries ...
    addPass<dawn::PassTemporaryType>(context, passes);
    addPass<dawn::PassFixVersionedInputFields>(context, passes);
    break;

  case PassGroup::PassTmpToFunction:
    addPass<dawn::PassTemporaryToStencilFunction>(context, passes);
    break;

  case PassGroup::SetNonTempCaches:
    addPass<dawn::PassSetNonTempCaches>(context, passes);
    // this should not affect the temporaries but since we're touching them it would probably be a
    // safe idea
    addPass<dawn::PassTemporaryType>(context, passes);
    break;

  case PassGroup::SetCaches:
    addPass<dawn::PassSetCaches>(context, passes);
    break;

  case PassGroup::SetBlockSize:
    addPass<dawn::PassSetBlockSize>(context, passes);
    break;

  case PassGroup::DataLocalityMetric:
    addPass<dawn::PassDataLocalityMetric>(context, passes);
    break;

  default:
    break;
  }

  return passes;
}

bool CompilerUtil::runGroup(PassGroup group, std::unique_ptr<OptimizerContext>& context) {
  bool result = true;
  for(auto& [name, instantiation] : context->getStencilInstantiationMap()) {
    result &= runGroup(group, context, instantiation);
  }
  return result;
}

bool CompilerUtil::runGroup(PassGroup group, std::unique_ptr<OptimizerContext>& context,
                            std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation) {
  auto mssSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy::Optimized;
  auto inlineStrategy = dawn::PassInlining::InlineStrategy::InlineProcedures;
  auto reorderStrategy = dawn::ReorderStrategy::Kind::Greedy;
  auto inlineOnTheFly = dawn::PassInlining::InlineStrategy::ComputationsOnTheFly;
  bool result = true;

  switch(group) {
  case PassGroup::Parallel:
    result &= runPass<dawn::PassInlining>(context, instantiation, true, inlineStrategy);
    result &= runPass<dawn::PassFieldVersioning>(context, instantiation);
    result &= runPass<dawn::PassSSA>(context, instantiation); // Did I do this for a reason?
    result &= runPass<dawn::PassMultiStageSplitter>(context, instantiation, mssSplitStrategy);
    result &= runPass<dawn::PassStageSplitter>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    result &= runPass<dawn::PassFixVersionedInputFields>(context, instantiation);
    result &= runPass<dawn::PassComputeStageExtents>(context, instantiation);
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);
    break;

  case PassGroup::ReorderStages:
    result &= runPass<dawn::PassSetStageGraph>(context, instantiation);
    result &= runPass<dawn::PassStageReordering>(context, instantiation, reorderStrategy);
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);
    result &= runPass<dawn::PassSetStageName>(context, instantiation);
    break;

  case PassGroup::MergeStages:
    result &= runPass<dawn::PassStageMerger>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    result &= runPass<dawn::PassFixVersionedInputFields>(context, instantiation);
    result &= runPass<dawn::PassComputeStageExtents>(context, instantiation);
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);
    break;

  case PassGroup::MergeTemporaries:
    result &= runPass<dawn::PassTemporaryMerger>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    break;

  case PassGroup::Inlining:
    result &= runPass<dawn::PassInlining>(context, instantiation, false, inlineOnTheFly);
    break;

  case PassGroup::PartitionIntervals:
    result &= runPass<dawn::PassIntervalPartitioner>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    result &= runPass<dawn::PassFixVersionedInputFields>(context, instantiation);
    break;

  case PassGroup::PassTmpToFunction:
    result &= runPass<dawn::PassTemporaryToStencilFunction>(context, instantiation);
    break;

  case PassGroup::SetNonTempCaches:
    result &= runPass<dawn::PassSetNonTempCaches>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    break;

  case PassGroup::SetCaches:
    result &= runPass<dawn::PassSetCaches>(context, instantiation);
    break;

  case PassGroup::SetBlockSize:
    result &= runPass<dawn::PassSetBlockSize>(context, instantiation);
    break;

  case PassGroup::DataLocalityMetric:
    result &= runPass<dawn::PassDataLocalityMetric>(context, instantiation);
    break;

  default:
    result = false;
    break;
  }
  return result;
}

void CompilerUtil::write(const std::shared_ptr<dawn::SIR>& sir, const std::string& filePrefix) {
  dawn::UIDGenerator::getInstance()->reset();
  if(Verbose)
    std::cerr << "Writing SIR file '" << filePrefix << ".sir'" << std::endl;
  dawn::SIRSerializer::serialize(filePrefix + ".sir", sir.get());
}

void CompilerUtil::write(std::unique_ptr<OptimizerContext>& context, const unsigned level,
                         const unsigned maxLevel, const std::string& filePrefix) {
  if(level > maxLevel)
    return;
  unsigned nstencils = context->getStencilInstantiationMap().size();
  unsigned stencil_id = 0;

  for(auto& [name, instantiation] : context->getStencilInstantiationMap()) {
    std::string iirFile = filePrefix;
    if(nstencils > 1)
      iirFile += "." + std::to_string(stencil_id);
    if(maxLevel > 0)
      iirFile += ".O" + std::to_string(level);
    iirFile += ".iir";

    if(Verbose)
      std::cerr << "Writing IIR file '" << iirFile << "'" << std::endl;
    dawn::IIRSerializer::serialize(iirFile, instantiation);
    stencil_id += 1;
  }
}

} // namespace dawn
