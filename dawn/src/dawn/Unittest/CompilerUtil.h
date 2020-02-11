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

#ifndef DAWN_UNITTEST_COMPILERUTIL_H
#define DAWN_UNITTEST_COMPILERUTIL_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/Compiler/DawnCompiler.h"
#include "dawn/Compiler/Options.h"
#include "dawn/Serialization/IIRSerializer.h"
#include "dawn/Serialization/SIRSerializer.h"

#include <fstream>
#include <iostream>

namespace dawn {

enum class PassGroup {
  Parallel,
  SSA,
  PrintStencilGraph,
  SetStageName,
  ReorderStages,
  MergeStages,
  MergeTemporaries,
  Inlining,
  PartitionIntervals,
  PassTmpToFunction,
  SetNonTempCaches,
  SetCaches,
  SetBlockSize,
  DataLocalityMetric
};

using stencilInstantiationContext =
    std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>;

/// @brief Compiler utilities for unit tests
/// @ingroup unittest
class CompilerUtil {
public:
  static bool Verbose;

  static std::shared_ptr<SIR> load(const std::string& sirFilename);
  static std::shared_ptr<iir::StencilInstantiation>
  load(const std::string& iirFilename,
       const dawn::OptimizerContext::OptimizerContextOptions& options,
       std::unique_ptr<OptimizerContext>& context, const std::string& envPath = "");
  static std::shared_ptr<iir::StencilInstantiation>
  lower(const std::shared_ptr<dawn::SIR>& sir,
        const dawn::OptimizerContext::OptimizerContextOptions& options,
        std::unique_ptr<OptimizerContext>& context);
  static std::shared_ptr<iir::StencilInstantiation>
  lower(const std::string& sirFilename,
        const dawn::OptimizerContext::OptimizerContextOptions& options,
        std::unique_ptr<OptimizerContext>& context, const std::string& envPath = "");
  static stencilInstantiationContext compile(const std::shared_ptr<SIR>& sir);
  static stencilInstantiationContext compile(const std::string& sirFile);
  static void clearDiags();
  static bool generate(const std::unique_ptr<OptimizerContext>& context,
                       const std::string& outFile = "");
  static void dumpNaive(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);
  static void dumpCuda(std::ostream& os, dawn::codegen::stencilInstantiationContext& ctx);

  template <class TPass, typename... Args>
  static void addPass(std::unique_ptr<OptimizerContext>& context,
                      std::vector<std::shared_ptr<Pass>>& passes, Args&&... args) {
    std::shared_ptr<Pass> pass = std::make_shared<TPass>(*context, std::forward<Args>(args)...);
    passes.emplace_back(pass);
  }

  template <class TPass, typename... Args>
  static bool runPass(std::unique_ptr<OptimizerContext>& context,
                      std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation,
                      Args&&... args) {
    TPass pass(*context, std::forward<Args>(args)...);
    if(Verbose)
      std::cerr << "Running pass: '" << pass.getName() << "' in stencil '"
                << instantiation->getName() << "'\n";
    return pass.run(instantiation);
  }

  static bool runPasses(std::unique_ptr<OptimizerContext>& context,
                        std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation,
                        unsigned nPasses = 100);

  static std::vector<std::shared_ptr<Pass>> createGroup(PassGroup group,
                                                        std::unique_ptr<OptimizerContext>& context);
  static bool runGroup(PassGroup group, std::unique_ptr<OptimizerContext>& context);
  static bool runGroup(PassGroup group, std::unique_ptr<OptimizerContext>& context,
                       std::shared_ptr<dawn::iir::StencilInstantiation>& instantiation);

  static void write(const std::shared_ptr<SIR>& sir, const std::string& filePrefix);

  static void write(std::unique_ptr<OptimizerContext>& context, const unsigned level = 0,
                    const unsigned maxLevel = 100, const std::string& filePrefix = "");

private:
  static dawn::DiagnosticsEngine diag_;
};

} // namespace dawn

#endif
