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

#include <sstream>
#include <unordered_set>

#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/Support/IndexRange.h"
#include "dawn/Support/Iterator.h"

namespace dawn {
namespace iir {
class StencilInstantiation;
}

namespace codegen {
namespace cuda {

/// @brief GridTools C++ code generation for the gtclang DSL
/// @ingroup cxxnaive
class MSCodeGen {
  struct KCacheProperties {
    inline KCacheProperties(std::string name, int accessID, iir::Extent intervalVertExtent)
        : name_(name), accessID_(accessID), intervalVertExtent_(intervalVertExtent) {}
    std::string name_;
    int accessID_;
    iir::Extent intervalVertExtent_; // extent of the cache used within the interval,
    // this information is used for IO policies, to know
    // which portion of the interval needs to be sync with mem
  };

private:
  std::stringstream& ss_;
  const std::unique_ptr<iir::MultiStage>& ms_;
  const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation_;
  const iir::StencilMetaInformation& metadata_;
  const CacheProperties& cacheProperties_;
  bool useCodeGenTemporaries_;
  std::string cudaKernelName_;
  Array3ui blockSize_;
  const bool solveKLoopInParallel_;
  CudaCodeGen::CudaCodeGenOptions options_;
  bool iterationSpaceSet_;
  static std::unordered_set<std::string> globalNames_;

public:
  MSCodeGen(std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
            const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
            const CacheProperties& cacheProperties, CudaCodeGen::CudaCodeGenOptions options,
            bool iterationSpaceSet = false);

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

  void generateKCacheFillStatement(MemberFunction& cudaKernel,
                                   const std::unordered_map<int, Array3i>& fieldIndexMap,
                                   const KCacheProperties& kcacheProp, int klev) const;

  static std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                       iir::Interval::Bound bound);
  std::string makeKLoop(const std::string dom, iir::Interval const& interval, bool kParallel);

  static std::string makeLoopImpl(const iir::Extent extent, const std::string& dim,
                                  const std::string& lower, const std::string& upper,
                                  const std::string& comparison, const std::string& increment);

  static std::string kBegin(const std::string dom, iir::LoopOrderKind loopOrder,
                            iir::Interval const& interval);

  /// @brief determines the multi interval of an interval (targetInterval) has not been accessed
  /// before the execution of the queryInterval by a given accessID
  iir::MultiInterval intervalNotPreviouslyAccessed(const int accessID,
                                                   const iir::Interval& targetInterval,
                                                   iir::Interval const& queryInterval) const;

  /// @brief returns true if the stage is the last stage of an interval loop execution
  /// which requires synchronization due to usage of 2D ij caches (which are re-written at the
  /// next
  /// k-loop iteration)
  bool intervalRequiresSync(const iir::Interval& interval, const iir::Stage& stage) const;

  /// @brief determines if a cache needs to flush for a given interval
  bool checkIfCacheNeedsToFlush(const iir::Cache& cache, iir::Interval interval) const;

  void generateFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                            const std::unordered_map<int, Array3i>& fieldIndexMap,
                            iir::Cache::IOPolicy policy) const;
  /// @brief computes additional information of kcaches for those kache with IO synchronization
  /// policy
  std::unordered_map<iir::Extents, std::vector<KCacheProperties>>
  buildKCacheProperties(const iir::Interval& interval, const iir::Cache::IOPolicy policy) const;

  /// @brief generates the kcache flush statement, that can be guarded by a conitional to protect
  /// for out-of-bounds or not, depending on the distance from the interval being executed to the
  /// interval range where cache is declared
  void generateKCacheFlushBlockStatement(MemberFunction& cudaKernel, const iir::Interval& interval,
                                         const std::unordered_map<int, Array3i>& fieldIndexMap,
                                         const KCacheProperties& kcacheProp, const int klev,
                                         std::string currentKLevel) const;

  /// @brief generates the kcache flush statement
  void generateKCacheFlushStatement(MemberFunction& cudaKernel,
                                    const std::unordered_map<int, Array3i>& fieldIndexMap,
                                    const int accessID, std::string cacheName,
                                    const int offset) const;

  /// @brief code generate slides of the values of a kcache in a ring-buffer manner
  void generateKCacheSlide(MemberFunction& cudaKernel, const iir::Interval& interval) const;

  /// @brief generates a final kcache flush statement, i.e. it flushes all the levels at the end of
  /// an interval iteration
  void generateFinalFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                 const std::unordered_map<int, Array3i>& fieldIndexMap,
                                 const iir::Cache::IOPolicy policy) const;
};

} // namespace cuda
} // namespace codegen
} // namespace dawn
