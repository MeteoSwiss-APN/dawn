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
#include "dawn/Optimizer/Pass.h"
#include "dawn/Support/NonCopyable.h"

#include <string>
#include <vector>

namespace gtclang {

/// @brief Emulate invocation of GTClang from command-line
/// @ingroup unittest
class IRSplitter : dawn::NonCopyable {
  std::unique_ptr<dawn::OptimizerContext> context_;
  dawn::DiagnosticsEngine diag_;
  std::string filePrefix_;

public:
  explicit IRSplitter();

  void split(const std::string& dslFile, const std::vector<std::string>& args = {});
  void parallelize();
  void optimize();
  void generate(const std::string& outFile = "");

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

  template <class T, typename... Args>
  bool runPass(const std::string& name,
               std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation, Args&&... args);
};

} // namespace gtclang

#endif
