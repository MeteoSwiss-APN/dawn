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
#include "dawn/SIR/SIR.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"
#include "dawn/Support/DiagnosticsEngine.h"
#include "dawn/Support/FileSystem.h"
#include "dawn/Unittest/CompilerUtil.h"
#include "gtclang/Unittest/Config.h"
#include "gtclang/Unittest/GTClang.h"
#include <fstream>

namespace gtclang {

using CompilerUtil = dawn::CompilerUtil;

IRSplitter::IRSplitter(const std::string& destDir, unsigned maxLevel)
    : filePrefix_(destDir), maxLevel_(maxLevel) {}

void IRSplitter::split(const std::string& dslFile, const std::vector<std::string>& args) {
  fs::path filePath(dslFile);
  if(filePrefix_.empty())
    filePrefix_ = filePath.root_directory().string();
  filePrefix_ += "/" + filePath.stem().string();

  std::vector<std::string> flags = {"-std=c++11",
                                    std::string{"-I"} + std::string{GTCLANG_UNITTEST_INCLUDES}};
  for(const auto& arg : args) {
    flags.emplace_back(arg);
  }

  auto [success, sir] = GTClang::run({dslFile, "-fno-codegen"}, flags);

  if(sir.get()) {
    CompilerUtil::Verbose = true;

    // Serialize the SIR
    CompilerUtil::write(sir, filePrefix_);

    // Use SIR to create context
    createContext(sir);

    // Lower to unoptimized IIR and serialize
    writeIIR();

    if(success && maxLevel_ > 0) {
      // Run parallelization passes
      parallelize();
      writeIIR(1);

      // Run optimization passes
      optimize();
    }
  }
}

void IRSplitter::createContext(const std::shared_ptr<dawn::SIR>& sir) {
  context_ = std::make_unique<dawn::OptimizerContext>(diag_, options_, sir);
}

void IRSplitter::parallelize() {
  CompilerUtil::runGroup(dawn::PassGroup::Parallel, context_);
}

void IRSplitter::optimize() {
  unsigned level = 1;

  // Reorder stages
  reorderStages();
  level += 1;
  writeIIR(level);

  // Merge stages
  mergeStages();
  level += 1;
  writeIIR(level);

  // Merge temporaries
  mergeTemporaries();
  level += 1;
  writeIIR(level);

  // Next inlining step
  inlining();
  level += 1;
  writeIIR(level);

  // Interval partitioning...
  partitionIntervals();
  level += 1;
  writeIIR(level);

  // Pass temporaries to functions
  // OFF by default (dawn/Optimizer/OptimizerOptions.inc)
  //  passTmpToFunction();
  //  level += 1;
  //  writeIIR(level);

  // OFF by default (dawn/Optimizer/OptimizerOptions.inc)
  //  setNonTempCaches();
  //  level += 1;
  //  writeIIR(level);

  // OFF by default (dawn/Optimizer/OptimizerOptions.inc)
    setCaches();
    level += 1;
    writeIIR(level);

  // Unsure whether this is ON by default -- probably only for CudaCodeGen
  //  setBlockSize();
  //  level += 1;
  //  writeIIR(level);

  // Unsure whether this is ON by default -- diagnostics only
  //  dataLocalityMetric();
  //  level += 1;
  //  writeIIR(level);
}

void IRSplitter::reorderStages() {
  CompilerUtil::runGroup(dawn::PassGroup::ReorderStages, context_);
}

void IRSplitter::mergeStages() {
  CompilerUtil::runGroup(dawn::PassGroup::MergeStages, context_);
}

void IRSplitter::mergeTemporaries() {
  CompilerUtil::runGroup(dawn::PassGroup::MergeTemporaries, context_);
}

void IRSplitter::inlining() {
  CompilerUtil::runGroup(dawn::PassGroup::Inlining, context_);
}

void IRSplitter::partitionIntervals() {
  CompilerUtil::runGroup(dawn::PassGroup::PartitionIntervals, context_);
}

void IRSplitter::passTmpToFunction() {
  CompilerUtil::runGroup(dawn::PassGroup::PassTmpToFunction, context_);
}

void IRSplitter::setNonTempCaches() {
  CompilerUtil::runGroup(dawn::PassGroup::SetNonTempCaches, context_);
}

void IRSplitter::setCaches() {
  CompilerUtil::runGroup(dawn::PassGroup::SetCaches, context_);
}

void IRSplitter::setBlockSize() {
  CompilerUtil::runGroup(dawn::PassGroup::SetBlockSize, context_);
}

void IRSplitter::dataLocalityMetric() {
  CompilerUtil::runGroup(dawn::PassGroup::DataLocalityMetric, context_);
}

void IRSplitter::writeIIR(const unsigned level) {
  CompilerUtil::write(context_, level, maxLevel_, filePrefix_);
}

} // namespace gtclang
