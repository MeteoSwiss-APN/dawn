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

#ifndef GTCLANG_UNITTEST_IRSPLITTER_H
#define GTCLANG_UNITTEST_IRSPLITTER_H

#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Support/NonCopyable.h"

#include <string>
#include <vector>

namespace gtclang {

/// @brief Split SIR into various levels of IIR
/// @ingroup unittest
class IRSplitter : dawn::NonCopyable {
  std::unique_ptr<dawn::OptimizerContext> context_;
  dawn::DiagnosticsEngine diag_;
  dawn::OptimizerContext::OptimizerContextOptions options_;
  std::string filePrefix_;
  unsigned maxLevel_;
  bool verbose_;

public:
  explicit IRSplitter(const std::string& destDir = "", unsigned maxLevel = 1000);
  void split(const std::string& dslFile, const std::vector<std::string>& args = {});
  void parallelize();
  void optimize();

protected:
  void createContext(const std::shared_ptr<dawn::SIR>& sir);
  void writeIIR(const unsigned level = 0);

  // Pass groups
  void reorderStages();
  void mergeStages();
  void mergeTemporaries();
  void inlining();
  void partitionIntervals();
  void passTmpToFunction();
  void setNonTempCaches();
  void setCaches();
  void setBlockSize();
  void dataLocalityMetric();
};

} // namespace gtclang

#endif
