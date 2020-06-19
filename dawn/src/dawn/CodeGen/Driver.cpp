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

#include "dawn/CodeGen/Driver.h"
#include "dawn/CodeGen/CXXNaive-ico/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/CXXNaive/CXXNaiveCodeGen.h"
#include "dawn/CodeGen/Cuda-ico/CudaIcoCodeGen.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/GridTools/GTCodeGen.h"
#include "dawn/Serialization/IIRSerializer.h"

#include <stdexcept>

namespace dawn {
namespace codegen {

codegen::Backend parseBackendString(const std::string& backendStr) {
  if(backendStr == "gt" || backendStr == "gridtools") {
    return codegen::Backend::GridTools;
  } else if(backendStr == "naive" || backendStr == "cxxnaive" || backendStr == "c++-naive") {
    return codegen::Backend::CXXNaive;
  } else if(backendStr == "ico" || backendStr == "naive-ico" || backendStr == "c++-naive-ico") {
    return codegen::Backend::CXXNaiveIco;
  } else if(backendStr == "cuda" || backendStr == "CUDA") {
    return codegen::Backend::CUDA;
  } else if(backendStr == "cuda-ico" || backendStr == "CUDAIco" || backendStr == "CUDA-Ico" ||
            backendStr == "CUDA-ICO") {
    return codegen::Backend::CUDAIco;
  } else {
    throw std::invalid_argument("Backend not supported");
  }
}

std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>& context,
    Backend backend, const Options& options) {
  switch(backend) {
  case Backend::CUDA:
    return cuda::run(context, options);
  case Backend::CXXNaive:
    return cxxnaive::run(context, options);
  case Backend::CXXNaiveIco:
    return cxxnaiveico::run(context, options);
  case Backend::GridTools:
    return gt::run(context, options);
  case Backend::CUDAIco:
    return cudaico::run(context, options);
  case Backend::CXXOpt:
    throw std::invalid_argument("Backend not supported");
  }
  // This line should not be needed but the compiler seems to complain if it is not present.
  return nullptr;
}

std::string run(const std::map<std::string, std::string>& stencilInstantiationMap,
                dawn::IIRSerializer::Format format, dawn::codegen::Backend backend,
                const dawn::codegen::Options& options) {
  std::map<std::string, std::shared_ptr<dawn::iir::StencilInstantiation>> internalMap;
  for(auto [name, instStr] : stencilInstantiationMap) {
    internalMap.insert(
        std::make_pair(name, dawn::IIRSerializer::deserializeFromString(instStr, format)));
  }
  return dawn::codegen::generate(dawn::codegen::run(internalMap, backend, options));
}

/// @brief Run code generation on a single stencil instantiation
std::unique_ptr<TranslationUnit>
run(const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation, Backend backend,
    const Options& options) {
  return run({{stencilInstantiation->getName(), stencilInstantiation}}, backend, options);
}

std::string generate(const std::unique_ptr<TranslationUnit>& translationUnit) {
  std::string code;
  for(const auto& p : translationUnit->getPPDefines())
    code += p + "\n";

  code += translationUnit->getGlobals() + "\n\n";
  for(const auto& p : translationUnit->getStencils())
    code += p.second;

  return code;
}

} // namespace codegen
} // namespace dawn
