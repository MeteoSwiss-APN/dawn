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

#include "dawn/CodeGen/Cuda-ico/CudaIcoCodeGen.h"

#include "dawn/Support/Logging.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cudaico {
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options) {
  DiagnosticsEngine diagnostics;
  const Array3i domain_size{options.DomainSizeI, options.DomainSizeJ, options.DomainSizeK};
  CudaIcoCodeGen CG(stencilInstantiationMap, diagnostics, options.MaxHaloSize, options.nsms,
                    options.MaxBlocksPerSM, domain_size);
  if(diagnostics.hasDiags()) {
    for(const auto& diag : diagnostics.getQueue())
      DAWN_LOG(INFO) << diag->getMessage();
    throw std::runtime_error("An error occured in code generation");
  }

  return CG.generateCode();
}

CudaIcoCodeGen::CudaIcoCodeGen(const StencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                               int maxHaloPoints, int nsms, int maxBlocksPerSM,
                               const Array3i& domainSize)
    : CodeGen(ctx, engine, maxHaloPoints) {}

CudaIcoCodeGen::~CudaIcoCodeGen() {}

std::unique_ptr<TranslationUnit> CudaIcoCodeGen::generateCode() {
  std::vector<std::string> ppDefines;
  std::map<std::string, std::string> stencils;

  std::cout << "in cuda ico codegen\n";

  return std::make_unique<TranslationUnit>("todo.txt", std::move(ppDefines), std::move(stencils),
                                           "");
}

} // namespace cudaico
} // namespace codegen
} // namespace dawn