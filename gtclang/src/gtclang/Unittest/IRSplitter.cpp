//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gtclang/Unittest/IRSplitter.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/Optimizer/PassComputeStageExtents.h"
#include "dawn/Optimizer/PassFieldVersioning.h"
#include "dawn/Optimizer/PassFixVersionedInputFields.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/Optimizer/PassMultiStageSplitter.h"
#include "dawn/Optimizer/PassSSA.h"
#include "dawn/Optimizer/PassSetStageGraph.h"
#include "dawn/Optimizer/PassSetStageName.h"
#include "dawn/Optimizer/PassSetSyncStage.h"
#include "dawn/Optimizer/PassStageMerger.h"
#include "dawn/Optimizer/PassStageReordering.h"
#include "dawn/Optimizer/PassStageSplitter.h"
#include "dawn/Optimizer/PassTemporaryType.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "gtclang/Unittest/Config.h"
#include "gtclang/Unittest/GTClang.h"
#include <fstream>

namespace gtclang {

IRSplitter::IRSplitter() {}

void IRSplitter::split(const std::string& dslFile) {
  filePrefix_ = dslFile;
  size_t pos = filePrefix_.rfind('.');
  if(pos != std::string::npos) {
    filePrefix_ = filePrefix_.substr(0, pos);
  }

  std::vector<std::string> flags = {"-std=c++11",
                                    std::string{"-I"} + std::string{GTCLANG_UNITTEST_INCLUDES}};
  std::pair<bool, std::shared_ptr<dawn::SIR>> tuple =
      GTClang::run({dslFile, "-fno-codegen"}, flags);

  if(tuple.first) {
    // Use SIR to create context then serialize the SIR
    std::shared_ptr<dawn::SIR> sir = tuple.second;
    createContext(sir);
    dawn::SIRSerializer::serialize(filePrefix_ + ".sir", sir.get());

    // Lower to unoptimized IIR and serialize
    unsigned level = 0;
    writeIIR(level);

    // Run parallelization passes
    parallelize();
    level += 1;
    writeIIR(level);

    // Reorder stages
    reorderStages();
    level += 1;
    writeIIR(level);

    // Merge stages
    mergeStages();
    level += 1;
    writeIIR(level);
  }
}

void IRSplitter::codegen(const std::string& outFile) {
  std::unique_ptr<dawn::codegen::TranslationUnit> tu;
  dawn::DiagnosticsEngine diagnostics;
  auto& ctx = context_->getStencilInstantiationMap();

  if(outFile.find(".cu") != std::string::npos) {
    dawn::codegen::cuda::CudaCodeGen generator(ctx, diagnostics, 0, 0, 0, {0, 0, 0});
    tu = generator.generateCode();
  } else {
    dawn::codegen::cxxnaive::CXXNaiveCodeGen generator(ctx, diagnostics, 0);
    tu = generator.generateCode();
  }

  std::ostringstream ss;
  for(auto const& macroDefine : tu->getPPDefines())
    ss << macroDefine << "\n";

  ss << tu->getGlobals();
  for(auto const& s : tu->getStencils())
    ss << s.second;

  if(outFile.empty()) {
    std::cerr << ss.str();
  } else {
    std::ofstream ofs(outFile.c_str());
    ofs << ss.str();
  }
}

void IRSplitter::createContext(std::shared_ptr<dawn::SIR>& sir) {
  dawn::DiagnosticsEngine diag;
  dawn::OptimizerContext::OptimizerContextOptions options;
  context_ = std::make_unique<dawn::OptimizerContext>(diag, options, sir);
}

void IRSplitter::parallelize() {
  dawn::PassInlining::InlineStrategy inlineStrategy =
      dawn::PassInlining::InlineStrategy::InlineProcedures;
  using MultistageSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy;
  // MultistageSplitStrategy mssSplitStrategy =  (options.MaxCutMSS) ?
  // MultistageSplitStrategy::MaxCut :
  MultistageSplitStrategy mssSplitStrategy = MultistageSplitStrategy::Optimized;

  for(auto& [name, instantiation] : context_->getStencilInstantiationMap()) {
    runPass<dawn::PassInlining>(name, instantiation, true, inlineStrategy);
    runPass<dawn::PassFieldVersioning>(name, instantiation);
    runPass<dawn::PassSSA>(name, instantiation);
    runPass<dawn::PassMultiStageSplitter>(name, instantiation, mssSplitStrategy);
    runPass<dawn::PassStageSplitter>(name, instantiation);
    runPass<dawn::PassTemporaryType>(name, instantiation);
    runPass<dawn::PassFixVersionedInputFields>(name, instantiation);
    runPass<dawn::PassComputeStageExtents>(name, instantiation);
    runPass<dawn::PassSetSyncStage>(name, instantiation);
  }
}

void IRSplitter::reorderStages() {
  // what is default?
  dawn::ReorderStrategy::Kind reorderStrategy = dawn::ReorderStrategy::Kind::Greedy;
  for(auto& [name, instantiation] : context_->getStencilInstantiationMap()) {
    runPass<dawn::PassSetStageGraph>(name, instantiation);
    runPass<dawn::PassStageReordering>(name, instantiation, reorderStrategy);
    runPass<dawn::PassSetSyncStage>(name, instantiation);
    runPass<dawn::PassSetStageName>(name, instantiation);
  }
}

void IRSplitter::mergeStages() {
  for(auto& [name, instantiation] : context_->getStencilInstantiationMap()) {
    runPass<dawn::PassStageMerger>(name, instantiation);
    // since this can change the scope of temporaries ...
    runPass<dawn::PassTemporaryType>(name, instantiation);
    runPass<dawn::PassFixVersionedInputFields>(name, instantiation);
    // modify stages and their extents ...
    runPass<dawn::PassComputeStageExtents>(name, instantiation);
    // and changes their dependencies
    runPass<dawn::PassSetSyncStage>(name, instantiation);
  }
}

template <class T, typename... Args>
bool IRSplitter::runPass(const std::string& name,
                         std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation,
                         Args&&... args) {
  T pass(*context_, std::forward<Args>(args)...);
  return pass.run(instantiation);
}

void IRSplitter::writeIIR(const unsigned level) {
  unsigned nstencils = 0;
  for(auto& [name, instantiation] : context_->getStencilInstantiationMap()) {
    dawn::IIRSerializer::serialize(filePrefix_ + "." + name + std::to_string(nstencils) +
                                   ".opt" + std::to_string(level) + ".iir", instantiation);
    nstencils += 1;
  }
}

} // namespace gtclang
