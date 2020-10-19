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

#include "LocToStringUtils.h"

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
  CudaIcoCodeGen(const StencilInstantiationContext& ctx, int maxHaloPoints,
                 std::optional<std::string> outputCHeader,
                 std::optional<std::string> outputFortranInterface);
  virtual ~CudaIcoCodeGen();
  virtual std::unique_ptr<TranslationUnit> generateCode() override;

  struct CudaIcoCodeGenOptions {
    // TODO: consider adding options for hard-coded values (e.g. BLOCK_SIZE)
    std::optional<std::string> OutputCHeader;
    std::optional<std::string> OutputFortranInterface;
  };

private:
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
  generateAllAPIRunFunctions(std::stringstream& ssSW,
                             const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                             CodeGenProperties& codeGenProperties, bool fromHost,
                             bool onlyDecl = false) const;

  void generateGpuMesh(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                       Class& stencilWrapperClass, CodeGenProperties& codeGenProperties);

  void generateRunFun(const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                      MemberFunction& runFun, CodeGenProperties& codeGenProperties);

  void generateGridFun(MemberFunction& runFun);

  void generateStencilClassCtr(MemberFunction& stencilClassCtor, const iir::Stencil& stencil,
                               const sir::GlobalVariableMap& globalsMap,
                               CodeGenProperties& codeGenProperties) const;

  void generateStencilClassCtrMinimal(MemberFunction& stencilClassCtor, const iir::Stencil& stencil,
                                      const sir::GlobalVariableMap& globalsMap,
                                      CodeGenProperties& codeGenProperties) const;  

  void generateStencilClassRawPtrCtr(MemberFunction& stencilClassCtor, const iir::Stencil& stencil,
                                     CodeGenProperties& codeGenProperties) const;

  void generateCopyBackFun(MemberFunction& copyBackFun, const iir::Stencil& stencil,
                           bool rawPtrs) const;

  void generateCopyMemoryFun(MemberFunction& copyFun, const iir::Stencil& stencil) const;

  void generateCopyPtrFun(MemberFunction& copyFun, const iir::Stencil& stencil) const;

  std::string generateStencilInstantiation(
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);

  void
  generateCHeaderSI(std::stringstream& ssSW,
                    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const;
  std::string generateCHeader() const;

  std::string generateF90Interface(std::string moduleName) const;

  CudaIcoCodeGenOptions codeGenOptions_;
};

} // namespace cudaico
} // namespace codegen
} // namespace dawn
