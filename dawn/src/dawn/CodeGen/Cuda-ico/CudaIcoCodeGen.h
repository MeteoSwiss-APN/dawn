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

#pragma once

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Options.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/IndexRange.h"
#include <unordered_map>

namespace dawn {
namespace iir {
class StencilInstantiation;
}

namespace codegen {
namespace cudaico {

/// @brief Run the Cuda code generation
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options = {});

/// @brief CUDA code generation for cartesian grids
/// @ingroup cxxnaive cartesian
class CudaIcoCodeGen : public CodeGen {

public:
  ///@brief constructor
  CudaIcoCodeGen(const StencilInstantiationContext& ctx, DiagnosticsEngine& engine,
                 int maxHaloPoints, int nsms, int maxBlocksPerSM, const Array3i& domainSize);
  virtual ~CudaIcoCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  struct CudaCodeGenOptions {
    int nsms;
    int maxBlocksPerSM;
    Array3i domainSize;
  };

private:
};

} // namespace cudaico
} // namespace codegen
} // namespace dawn
