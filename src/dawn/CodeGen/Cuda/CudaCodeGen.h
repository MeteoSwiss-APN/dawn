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
#include "dawn/CodeGen/Cuda/CacheProperties.h"
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

  std::unordered_map<int, CacheProperties> cachePropertyMap_;

public:
  ///@brief constructor
  CudaCodeGen(OptimizerContext* context);
  virtual ~CudaCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

private:
  static std::string
  buildCudaKernelName(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                      const std::unique_ptr<iir::MultiStage>& ms);

  void addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const override;

  void addTmpStorageInit(
      MemberFunction& ctr, iir::Stencil const& stencil,
      IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const override;

  void
  generateCudaKernelCode(std::stringstream& ssSW,
                         const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                         const std::unique_ptr<iir::MultiStage>& ms,
                         const CacheProperties& cacheProperties);
  void
  generateAllCudaKernels(std::stringstream& ssSW,
                         const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);

  void
  generateStencilRunMethod(Structure& stencilClass, const iir::Stencil& stencil,
                           const CodeGenProperties& codeGenProperties,
                           const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const std::unordered_map<std::string, std::string>& paramNameToType,
                           const sir::GlobalVariableMap& globalsMap) const;

  void
  generateStencilClasses(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                         Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) const;
  void
  generateStencilWrapperCtr(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const CodeGenProperties& codeGenProperties) const;
  void generateStencilWrapperMembers(
      Class& stencilWrapperClass,
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
      CodeGenProperties& codeGenProperties) const;

  void generateStencilWrapperSyncMethod(Class& stencilWrapperClass) const;

  void
  generateStencilWrapperRun(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const CodeGenProperties& codeGenProperties) const;

  void
  generateStencilWrapperPublicMemberFunctions(Class& stencilWrapperClass,
                                              const CodeGenProperties& codeGenProperties) const;

  void
  generateStencilClassCtr(Structure& stencilClass, const iir::Stencil& stencil,
                          const sir::GlobalVariableMap& globalsMap,
                          IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
                          IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
                          std::shared_ptr<StencilProperties> stencilProperties) const;

  void generateStencilClassMembers(
      Structure& stencilClass, const iir::Stencil& stencil,
      const sir::GlobalVariableMap& globalsMap,
      IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& nonTempFields,
      IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields,
      std::shared_ptr<StencilProperties> stencilProperties) const;

  std::string generateStencilInstantiation(
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);
};
} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
