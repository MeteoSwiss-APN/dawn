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
#include "dawn/Optimizer/PassIntervalPartitioning.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassPrintStencilGraph.h"
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
#include "dawn/Optimizer/PassStencilSplitter.h"
#include "dawn/Optimizer/PassTemporaryMerger.h"
#include "dawn/Optimizer/PassTemporaryToStencilFunction.h"
#include "dawn/Optimizer/PassTemporaryType.h"

#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"

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
  std::string filename = irFilename;
  if(!envPath.empty() && filename.at(0) != '/')
    filename = envPath + "/" + filename;

  if(filename.find(".sir") != std::string::npos) {
    stencilInstantiationContext siMap = lower(filename, options, context, envPath);
    return siMap.begin()->second;
  } else {
    std::shared_ptr<SIR> sir = std::make_shared<SIR>(ast::GridType::Cartesian);
    context = std::make_unique<OptimizerContext>(diag_, options, sir);
    return IIRSerializer::deserialize(filename);
  }
}

stencilInstantiationContext
CompilerUtil::lower(const std::shared_ptr<dawn::SIR>& sir,
                    const dawn::OptimizerContext::OptimizerContextOptions& options,
                    std::unique_ptr<OptimizerContext>& context) {
  context = std::make_unique<OptimizerContext>(diag_, options, sir);
  return context->getStencilInstantiationMap();
}

stencilInstantiationContext
CompilerUtil::lower(const std::string& sirFilename,
                    const dawn::OptimizerContext::OptimizerContextOptions& options,
                    std::unique_ptr<OptimizerContext>& context, const std::string& envPath) {
  std::string filename = sirFilename;
  if(!envPath.empty() && filename.at(0) != '/')
    filename = envPath + "/" + filename;

  std::shared_ptr<dawn::SIR> sir = load(filename);
  return lower(sir, options, context);
}

stencilInstantiationContext CompilerUtil::compile(const std::shared_ptr<SIR>& sir) {
  dawn::Options options;
  DawnCompiler compiler(options);

  auto SI = compiler.optimize(compiler.lowerToIIR(sir));

  if(compiler.getDiagnostics().hasDiags()) {
    for(const auto& diag : compiler.getDiagnostics().getQueue()) {
      std::cerr << "Compilation Error " << diag->getMessage() << std::endl;
    }
    throw std::runtime_error("Compilation failed");
  }

  return SI;
}

stencilInstantiationContext CompilerUtil::compile(const std::string& sirFile) {
  std::shared_ptr<SIR> sir = load(sirFile);
  return compile(sir);
}

void CompilerUtil::clearDiags() { diag_.clear(); }

namespace {
template <typename CG>
void dump(CG& generator, std::ostream& os) {
  auto tu = generator.generateCode();

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;
  os << ss.str();
}

dawn::codegen::stencilInstantiationContext
siToContext(std::shared_ptr<iir::StencilInstantiation> si) {
  dawn::codegen::stencilInstantiationContext ctx;
  ctx[si->getName()] = si;
  return ctx;
}

} // namespace

void CompilerUtil::dumpNaive(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si) {
  dawn::DiagnosticsEngine diagnostics;
  auto ctx = siToContext(si);
  dawn::codegen::cxxnaive::CXXNaiveCodeGen generator(ctx, diagnostics, 0);
  dump(generator, os);
}

void CompilerUtil::dumpNaiveIco(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si) {
  dawn::DiagnosticsEngine diagnostics;
  auto ctx = siToContext(si);
  dawn::codegen::cxxnaiveico::CXXNaiveIcoCodeGen generator(ctx, diagnostics, 0);
  dump(generator, os);
}

void CompilerUtil::dumpCuda(std::ostream& os, std::shared_ptr<iir::StencilInstantiation> si) {
  dawn::DiagnosticsEngine diagnostics;
  auto ctx = siToContext(si);
  dawn::codegen::cuda::CudaCodeGen generator(ctx, diagnostics, 0, 0, 0, {0, 0, 0});
  dump(generator, os);
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
    addPass<dawn::PassIntervalPartitioning>(context, passes);
    // since this can change the scope of temporaries ...
    addPass<dawn::PassTemporaryType>(context, passes);
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
    addPass<dawn::PassFixVersionedInputFields>(context, passes);
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

bool CompilerUtil::runPasses(unsigned nPasses, std::unique_ptr<OptimizerContext>& context,
                             std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation) {
  auto mssSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy::Optimized;
  auto inlineStrategy = dawn::PassInlining::InlineStrategy::InlineProcedures;
  auto reorderStrategy = dawn::ReorderStrategy::Kind::Greedy;
  auto inlineOnTheFly = dawn::PassInlining::InlineStrategy::ComputationsOnTheFly;

  bool result = true;
  if(nPasses > 0)
    result &= runPass<dawn::PassInlining>(context, instantiation, true, inlineStrategy);
  if(nPasses > 1)
    result &= runPass<dawn::PassFieldVersioning>(context, instantiation);
  if(nPasses > 2)
    result &= runPass<dawn::PassSSA>(context, instantiation);
  if(nPasses > 3)
    result &= runPass<dawn::PassMultiStageSplitter>(context, instantiation, mssSplitStrategy);
  if(nPasses > 4)
    result &= runPass<dawn::PassStageSplitter>(context, instantiation);
  if(nPasses > 5)
    result &= runPass<dawn::PassPrintStencilGraph>(context, instantiation);
  if(nPasses > 6)
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
  if(nPasses > 7)
    result &= runPass<dawn::PassSetStageName>(context, instantiation);
  if(nPasses > 8)
    result &= runPass<dawn::PassSetStageGraph>(context, instantiation);
  if(nPasses > 9)
    result &= runPass<dawn::PassStageReordering>(context, instantiation, reorderStrategy);
  if(nPasses > 10)
    result &= runPass<dawn::PassStageMerger>(context, instantiation);
  if(nPasses > 11)
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
  if(nPasses > 12)
    result &= runPass<dawn::PassTemporaryMerger>(context, instantiation);
  if(nPasses > 13)
    result &= runPass<dawn::PassInlining>(context, instantiation, false, inlineOnTheFly);
  if(nPasses > 14)
    result &= runPass<dawn::PassIntervalPartitioning>(context, instantiation);
  if(nPasses > 15)
    result &= runPass<dawn::PassTemporaryToStencilFunction>(context, instantiation);
  if(nPasses > 16)
    result &= runPass<dawn::PassSetNonTempCaches>(context, instantiation);
  if(nPasses > 17)
    result &= runPass<dawn::PassSetCaches>(context, instantiation);
  if(nPasses > 18)
    result &= runPass<dawn::PassFixVersionedInputFields>(context, instantiation);
  if(nPasses > 19)
    result &= runPass<dawn::PassComputeStageExtents>(context, instantiation);
  if(nPasses > 20) // if(getOptions().Backend == "cuda") {
    result &= runPass<dawn::PassSetBlockSize>(context, instantiation);
  if(nPasses > 21)
    result &= runPass<dawn::PassDataLocalityMetric>(context, instantiation);
  if(nPasses > 22)
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);

  return result;
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
    result &= runPass<dawn::PassSSA>(context, instantiation);
    result &= runPass<dawn::PassMultiStageSplitter>(context, instantiation, mssSplitStrategy);
    result &= runPass<dawn::PassStageSplitter>(context, instantiation);
    result &= runPass<dawn::PassPrintStencilGraph>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
    result &= runPass<dawn::PassComputeStageExtents>(context, instantiation);
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);
    break;

  case PassGroup::ReorderStages:
    result &= runPass<dawn::PassSetStageName>(context, instantiation);
    result &= runPass<dawn::PassSetStageGraph>(context, instantiation);
    result &= runPass<dawn::PassStageReordering>(context, instantiation, reorderStrategy);
    result &= runPass<dawn::PassSetSyncStage>(context, instantiation);
    result &= runPass<dawn::PassSetStageName>(context, instantiation);
    break;

  case PassGroup::MergeStages:
    result &= runPass<dawn::PassStageMerger>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
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
    result &= runPass<dawn::PassIntervalPartitioning>(context, instantiation);
    result &= runPass<dawn::PassTemporaryType>(context, instantiation);
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
