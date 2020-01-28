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

namespace gtclang {

void IRSplitter::split(const std::string& dslFile) {
  std::string fileStub = dslFile;
  size_t pos = fileStub.rfind('.');
  if(pos != std::string::npos) {
    fileStub = fileStub.substr(0, pos);
  }

  std::vector<std::string> flags = {"-std=c++11", "-I./src"};
  dawn::UIDGenerator::getInstance()->reset();
  std::pair<bool, std::shared_ptr<dawn::SIR>> tuple =
    GTClang::run({dslFile, "-fno-codegen"}, flags);

  if(tuple.first) {
    // Serialize the SIR
    std::shared_ptr<dawn::SIR> sir = tuple.second;
    dawn::SIRSerializer::serialize(fileStub + ".sir", sir.get());

    // Now lower to IIR
    dawn::DiagnosticsEngine diag;
    dawn::OptimizerContext::OptimizerContextOptions options;
    dawn::OptimizerContext context(diag, options, sir);
    context.fillIIR();

    // Serialize unoptimized IIR
    unsigned nstencils = 0;
    for(auto& [name, instantiation] : context.getStencilInstantiationMap()) {
      dawn::IIRSerializer::serialize(fileStub + ".unopt." + std::to_string(nstencils) + ".iir", instantiation);
      nstencils += 1;
    }

    // Run parallelization passes
      using MultistageSplitStrategy = dawn::PassMultiStageSplitter::MultiStageSplittingStrategy;
      //MultistageSplitStrategy mssSplitStrategy =  (options.MaxCutMSS) ? MultistageSplitStrategy::MaxCut :
      MultistageSplitStrategy mssSplitStrategy = MultistageSplitStrategy::Optimized;

      nstencils = 0;
      for(auto& [name, instantiation] : context.getStencilInstantiationMap()) {
        dawn::PassInlining(context, true, dawn::PassInlining::InlineStrategy::InlineProcedures).run(instantiation);
        dawn::PassFieldVersioning(context).run(instantiation);
        dawn::PassMultiStageSplitter(context, mssSplitStrategy).run(instantiation);
        dawn::PassStageSplitter(context).run(instantiation);
        dawn::PassTemporaryType(context).run(instantiation);
        dawn::PassFixVersionedInputFields(context).run(instantiation);
        dawn::PassComputeStageExtents(context).run(instantiation);
        dawn::PassSetSyncStage(context).run(instantiation);

        // Serialize parallelized stencil
        dawn::IIRSerializer::serialize(fileStub + ".par." + std::to_string(nstencils) + ".iir", instantiation);
        nstencils += 1;
      }

      // Run remaining passes...

      // Codegen...
  }
}

} // namespace gtclang
