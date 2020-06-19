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
namespace cuda {

/// @brief Run the Cuda code generation
std::unique_ptr<TranslationUnit>
run(const std::map<std::string, std::shared_ptr<iir::StencilInstantiation>>&
        stencilInstantiationMap,
    const Options& options = {});

/// @brief CUDA code generation for cartesian grids
/// @ingroup cxxnaive cartesian
class CudaCodeGen : public CodeGen {
  std::unordered_map<int, CacheProperties> cachePropertyMap_;

public:
  ///@brief constructor
  CudaCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints, int nsms,
              int maxBlocksPerSM, const Array3i& domainSize, bool runWithSync = true);
  virtual ~CudaCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  struct CudaCodeGenOptions {
    int nsms;
    int maxBlocksPerSM;
    Array3i domainSize;
    bool runWithSync;
  };

private:
  static std::string
  buildCudaKernelName(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                      const std::unique_ptr<iir::MultiStage>& ms);

  void addTempStorageTypedef(Structure& stencilClass, iir::Stencil const& stencil) const override;

  void addTmpStorageInit(
      MemberFunction& ctr, iir::Stencil const& stencil,
      IndexRange<const std::map<int, iir::Stencil::FieldInfo>>& tempFields) const override;

  void addCudaCopySymbol(MemberFunction& runMethod, const std::string& arrName,
                         const std::string dataType) const;

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
                           const std::shared_ptr<StencilProperties>& stencilProperties,
                           const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const std::unordered_map<std::string, std::string>& paramNameToType,
                           const sir::GlobalVariableMap& globalsMap) const;

  void
  generateStencilClasses(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                         Class& stencilWrapperClass, CodeGenProperties& codeGenProperties);
  void
  generateStencilWrapperCtr(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const CodeGenProperties& codeGenProperties) const;
  void generateStencilWrapperMembers(
      Class& stencilWrapperClass,
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
      CodeGenProperties& codeGenProperties) const;

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

  CudaCodeGenOptions codeGenOptions_;
  bool iterationSpaceSet_;
};

} // namespace cuda
} // namespace codegen
} // namespace dawn
