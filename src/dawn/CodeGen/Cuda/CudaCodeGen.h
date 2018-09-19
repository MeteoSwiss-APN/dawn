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

#ifndef DAWN_CODEGEN_CUDA_CUDACODEGEN_H
#define DAWN_CODEGEN_CUDA_CUDACODEGEN_H

#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/IIR/Interval.h"
#include "dawn/Support/IndexRange.h"
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dawn {
namespace iir {
class StencilInstantiation;
}

class OptimizerContext;

namespace codegen {
namespace cuda {

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup cxxnaive
class CudaCodeGen : public CodeGen {
public:
  ///@brief constructor
  CudaCodeGen(OptimizerContext* context);
  virtual ~CudaCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

private:
  static std::string buildCudaKernelName(const iir::StencilInstantiation* instantiation,
                                         const std::unique_ptr<iir::MultiStage>& ms);

  //  static std::string buildCudaKernelName(Class stencilWrapperClass, Structure& stencil);

  void addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const override;

  iir::Extents computeTempMaxWriteExtent(iir::Stencil const& stencil) const;

  void addTmpStorageInit(MemberFunction& ctr, iir::Stencil const& stencil,
                         IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>&
                             tempFields) const override;

  void generateCudaKernelCode(std::stringstream& ssSW,
                              const iir::StencilInstantiation* stencilInstantiation,
                              const std::unique_ptr<iir::MultiStage>& ms);
  void generateAllCudaKernels(std::stringstream& ssSW,
                              const iir::StencilInstantiation* stencilInstantiation);

  void generateRunMethod(Structure& stencilClass, const iir::Stencil& stencil,
                         const iir::StencilInstantiation* stencilInstantiation,
                         const std::unordered_map<int, std::string>& stencilTemplates,
                         const sir::GlobalVariableMap& globalsMap) const;

  std::string generateStencilInstantiation(const iir::StencilInstantiation* stencilInstantiation);
  std::string generateGlobals(const std::shared_ptr<SIR>& sir);
  static int paddedBoundary(int value);
};
} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
