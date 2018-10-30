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

  enum class FunctionArgType { caller, callee };
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

  iir::Extents computeTempMaxWriteExtent(iir::Stencil const& stencil) const;

  void addTmpStorageInit(MemberFunction& ctr, iir::Stencil const& stencil,
                         IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>&
                             tempFields) const override;

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
                           const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const std::unordered_map<std::string, std::string>& paramNameToType,
                           const sir::GlobalVariableMap& globalsMap) const;

  std::vector<std::string> generateStrideArguments(
      const IndexRange<const std::unordered_map<int, iir::Field>>& nonTempFields,
      const IndexRange<const std::unordered_map<int, iir::Field>>& tempFields,
      const iir::MultiStage& ms, const iir::StencilInstantiation& stencilInstantiation,
      FunctionArgType funArg) const;

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

  void
  generateStencilWrapperRun(Class& stencilWrapperClass,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const CodeGenProperties& codeGenProperties) const;

  void generateStencilWrapperSyncMethod(
      Class& stencilWrapperClass,
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
      const CodeGenProperties& codeGenProperties) const;

  void
  generateStencilWrapperPublicMemberFunctions(Class& stencilWrapperClass,
                                              const CodeGenProperties& codeGenProperties) const;

  void generateStencilClassCtr(
      Structure& stencilClass, const iir::Stencil& stencil,
      const sir::GlobalVariableMap& globalsMap,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& nonTempFields,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields,
      std::shared_ptr<StencilProperties> stencilProperties) const;

  void generateStencilClassMembers(
      Structure& stencilClass, const iir::Stencil& stencil,
      const sir::GlobalVariableMap& globalsMap,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& nonTempFields,
      IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields,
      std::shared_ptr<StencilProperties> stencilProperties) const;

  void generateIJCacheDecl(MemberFunction& kernel,
                           const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                           const iir::MultiStage& ms, const CacheProperties& cacheProperties,
                           Array3ui blockSize) const;

  void generateKCacheDecl(MemberFunction& kernel, const std::unique_ptr<iir::MultiStage>& ms,
                          const CacheProperties& cacheProperties) const;

  void generateIJCacheIndexInit(MemberFunction& kernel, const CacheProperties& cacheProperties,
                                const Array3ui blockSize) const;

  bool useTmpIndex(const std::unique_ptr<iir::MultiStage>& ms,
                   const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                   const CacheProperties& cacheProperties) const;

  void generateTmpIndexInit(MemberFunction& kernel, const std::unique_ptr<iir::MultiStage>& ms,
                            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                            const CacheProperties& cacheProperties) const;
  void generateKCacheSlide(MemberFunction& cudaKernel, const CacheProperties& cacheProperties,
                           const std::unique_ptr<iir::MultiStage>& ms,
                           const iir::Interval& interval) const;

  void
  generateFillKCaches(MemberFunction& cudaKernel, const std::unique_ptr<iir::MultiStage>& ms,
                      const iir::Interval& interval, const CacheProperties& cacheProperties,
                      const std::unordered_map<int, Array3i>& fieldIndexMap,
                      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const;

  void generatePreFillKCaches(
      MemberFunction& cudaKernel, const std::unique_ptr<iir::MultiStage>& ms,
      const iir::Interval& interval, const CacheProperties& cacheProperties,
      const std::unordered_map<int, Array3i>& fieldIndexMap,
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const;

  std::string intervalDiffToString(iir::IntervalDiff intervalDiff, std::string maxRange) const;

  iir::Interval::IntervalLevel computeNextLevelToProcess(const iir::Interval& interval,
                                                         iir::LoopOrderKind loopOrder) const;

  std::string generateStencilInstantiation(
      const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation);
  static int paddedBoundary(int value);
  /// @brief returns true if the stage is the last stage of an interval loop execution
  /// which requires synchronization due to usage of 2D ij caches (which are re-written at the next
  /// k-loop iteration)
  bool intervalRequiresSync(const iir::Interval& interval, const iir::Stage& stage,
                            const std::unique_ptr<iir::MultiStage>& ms) const;
};
} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
