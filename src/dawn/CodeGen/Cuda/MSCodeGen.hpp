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

#ifndef DAWN_CODEGEN_CUDA_MSCODEGEN_H
#define DAWN_CODEGEN_CUDA_MSCODEGEN_H

#include <sstream>

#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/Support/IndexRange.h"

namespace dawn {
namespace iir {
class StencilInstantiation;
}

namespace codegen {
namespace cuda {

namespace impl_ {
struct KCacheProperties {
  inline KCacheProperties(std::string name, int accessID, iir::Extent vertExtent)
      : name_(name), accessID_(accessID), vertExtent_(vertExtent) {}
  std::string name_;
  int accessID_;
  iir::Extent vertExtent_;
};
}

/// @brief GridTools C++ code generation for the gridtools_clang DSL
/// @ingroup cxxnaive
class MSCodeGen {
private:
  std::stringstream& ss_;
  const std::unique_ptr<iir::MultiStage>& ms_;
  const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation_;
  const CacheProperties& cacheProperties_;
  bool useTmpIndex_;
  std::string cudaKernelName_;
  Array3ui blockSize_;
  const bool solveKLoopInParallel_;

public:
  MSCodeGen(std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
            const CacheProperties& cacheProperties);

  void generateCudaKernelCode();

private:
  std::vector<std::string> generateStrideArguments(
      const IndexRange<const std::unordered_map<int, iir::Field>>& nonTempFields,
      const IndexRange<const std::unordered_map<int, iir::Field>>& tempFields,
      CodeGeneratorHelper::FunctionArgType funArg) const;

  /// @brief generate all IJ cache declarations
  void generateIJCacheDecl(MemberFunction& kernel) const;

  /// @brief code generate all kcache declarations
  void generateKCacheDecl(MemberFunction& kernel) const;
  static int paddedBoundary(int value);
  /// @brief code generate the ij cache index initialization
  void generateIJCacheIndexInit(MemberFunction& kernel) const;
  /// @brief code generate the initialization of a temporary field iterator
  void generateTmpIndexInit(MemberFunction& kernel) const;

  /// @brief return the first level that will initiate the interval processing, given a loop order
  iir::Interval::IntervalLevel computeNextLevelToProcess(const iir::Interval& interval,
                                                         iir::LoopOrderKind loopOrder) const;

  static std::string intervalDiffToString(iir::IntervalDiff intervalDiff, std::string maxRange);

  static std::string makeIntervalLevelBound(const std::string dom,
                                            iir::Interval::IntervalLevel const& intervalLevel);
  /// @brief generate a pre-fill of the kcaches, i.e. it fills all the klevels of the kcache that
  /// need to be filled before we start the k looping
  void generatePreFillKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                              const std::unordered_map<int, Array3i>& fieldIndexMap) const;
  /// @brief generate a fill of the top level of the kcache, at every k iteration
  void generateFillKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                           const std::unordered_map<int, Array3i>& fieldIndexMap) const;

  static std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                       iir::Interval::Bound bound);
  std::string makeKLoop(const std::string dom, iir::Interval const& interval, bool kParallel);

  static std::string makeLoopImpl(const iir::Extent extent, const std::string& dim,
                                  const std::string& lower, const std::string& upper,
                                  const std::string& comparison, const std::string& increment);

  static std::string kBegin(const std::string dom, iir::LoopOrderKind loopOrder,
                            iir::Interval const& interval);

  /// @brief returns true if the stage is the last stage of an interval loop execution
  /// which requires synchronization due to usage of 2D ij caches (which are re-written at the
  /// next
  /// k-loop iteration)
  bool intervalRequiresSync(const iir::Interval& interval, const iir::Stage& stage) const;

  void generateFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                            const std::unordered_map<int, Array3i>& fieldIndexMap) const;
  /// @brief computes additional information of kcaches for those kache with IO synchronization
  /// policy
  std::unordered_map<iir::Extents, std::vector<impl_::KCacheProperties>>
  buildKCacheProperties(const iir::Interval& interval, const iir::Cache::CacheIOPolicy policy,
                        const bool checkStrictIntervalBound) const;

  void generateKCacheFlushBlockStatement(MemberFunction& cudaKernel, const iir::Interval& interval,
                                         const std::unordered_map<int, Array3i>& fieldIndexMap,
                                         const impl_::KCacheProperties& kcacheProp, const int klev,
                                         std::string currentKLevel) const;

  void generateKCacheFlushStatement(MemberFunction& cudaKernel,
                                    const std::unordered_map<int, Array3i>& fieldIndexMap,
                                    const int accessID, std::string cacheName,
                                    const int offset) const;

  /// @brief code generate slides of the values of a kcache in a ring-buffer manner
  void generateKCacheSlide(MemberFunction& cudaKernel, const iir::Interval& interval) const;

  void generateFinalFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                 const std::unordered_map<int, Array3i>& fieldIndexMap,
                                 const iir::Cache::CacheIOPolicy policy) const;
};

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
