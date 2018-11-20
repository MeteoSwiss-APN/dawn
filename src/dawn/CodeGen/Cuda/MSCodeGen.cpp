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

#include "dawn/CodeGen/Cuda/MSCodeGen.hpp"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/Support/IndexRange.h"
#include <functional>
#include <numeric>

namespace dawn {
namespace codegen {
namespace cuda {
MSCodeGen::MSCodeGen(std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
                     const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                     const CacheProperties& cacheProperties)
    : ss_(ss), ms_(ms), stencilInstantiation_(stencilInstantiation),
      cacheProperties_(cacheProperties),
      blockSize_(stencilInstantiation_->getIIR()->getBlockSize()),
      solveKLoopInParallel_(CodeGeneratorHelper::solveKLoopInParallel(ms_)) {

  // useTmpIndex_
  const auto& fields = ms_->getFields();
  const bool containsTemporary =
      (find_if(fields.begin(), fields.end(), [&](const std::pair<int, iir::Field>& field) {
         const int accessID = field.second.getAccessID();
         if(!stencilInstantiation_->isTemporaryField(accessID))
           return false;
         // we dont need to initialize tmp indices for fields that are cached
         if(!cacheProperties_.accessIsCached(accessID))
           return true;
         const auto& cache = ms_->getCache(accessID);
         if(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local) {
           return false;
         }
         return true;
       }) != fields.end());

  useTmpIndex_ = containsTemporary && !CodeGeneratorHelper::useNormalIteratorForTmp(ms_);

  cudaKernelName_ = CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation_, ms_);
}

void MSCodeGen::generateIJCacheDecl(MemberFunction& kernel) const {
  for(const auto& cacheP : ms_->getCaches()) {
    const iir::Cache& cache = cacheP.second;
    if(cache.getCacheType() != iir::Cache::CacheTypeKind::IJ)
      continue;
    DAWN_ASSERT(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local);

    const int accessID = cache.getCachedFieldAccessID();
    const auto& maxExtents = cacheProperties_.getCacheExtent(accessID);

    kernel.addStatement(
        "__shared__ gridtools::clang::float_type " + cacheProperties_.getCacheName(accessID) + "[" +
        std::to_string(blockSize_[0] + (maxExtents[0].Plus - maxExtents[0].Minus)) + "*" +
        std::to_string(blockSize_[1] + (maxExtents[1].Plus - maxExtents[1].Minus)) + "]");
  }
}

void MSCodeGen::generateKCacheDecl(MemberFunction& kernel) const {
  for(const auto& cacheP : ms_->getCaches()) {
    const iir::Cache& cache = cacheP.second;

    if(cache.getCacheType() != iir::Cache::CacheTypeKind::K ||
       ((cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::local) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::fill) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::flush) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::epflush)))
      continue;

    if(cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::local && solveKLoopInParallel_)
      continue;
    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = cacheProperties_.getKCacheVertExtent(accessID);

    kernel.addStatement("gridtools::clang::float_type " + cacheProperties_.getCacheName(accessID) +
                        "[" + std::to_string(-vertExtent.Minus + vertExtent.Plus + 1) + "]");
  }
}

int MSCodeGen::paddedBoundary(int value) {
  return std::abs(value) <= 1 ? 1 : std::abs(value) <= 2 ? 2 : std::abs(value) <= 4 ? 4 : 8;
}

void MSCodeGen::generateIJCacheIndexInit(MemberFunction& kernel) const {
  if(cacheProperties_.isThereACommonCache()) {
    kernel.addStatement(
        "int " + cacheProperties_.getCommonCacheIndexName(iir::Cache::CacheTypeKind::IJ) +
        "= iblock + " + std::to_string(cacheProperties_.getOffsetCommonIJCache(1)) +
        " + (jblock + " + std::to_string(cacheProperties_.getOffsetCommonIJCache(1)) + ")*" +
        std::to_string(cacheProperties_.getStrideCommonCache(1, blockSize_)));
  }
}

iir::Interval::IntervalLevel
MSCodeGen::computeNextLevelToProcess(const iir::Interval& interval,
                                     iir::LoopOrderKind loopOrder) const {
  iir::Interval::IntervalLevel intervalLevel;
  if(loopOrder == iir::LoopOrderKind::LK_Backward) {
    intervalLevel = interval.upperIntervalLevel();
  } else {
    intervalLevel = interval.lowerIntervalLevel();
  }
  return intervalLevel;
}

void MSCodeGen::generateTmpIndexInit(MemberFunction& kernel) const {

  if(!useTmpIndex_)
    return;

  auto maxExtentTmps = CodeGeneratorHelper::computeTempMaxWriteExtent(*(ms_->getParent()));
  kernel.addStatement("int idx_tmp = (iblock+" + std::to_string(-maxExtentTmps[0].Minus) +
                      ")*1 + (jblock+" + std::to_string(-maxExtentTmps[1].Minus) + ")*jstride_tmp");
}

std::string MSCodeGen::intervalDiffToString(iir::IntervalDiff intervalDiff, std::string maxRange) {
  if(intervalDiff.rangeType_ == iir::IntervalDiff::RangeType::fullRange) {
    return maxRange + "+" + std::to_string(intervalDiff.value);
  }
  return std::to_string(intervalDiff.value);
}

std::string MSCodeGen::makeIntervalLevelBound(const std::string dom,
                                              iir::Interval::IntervalLevel const& intervalLevel) {
  return intervalLevel.isEnd() ? " ksize - 1 + " + std::to_string(intervalLevel.offset_)
                               : std::to_string(intervalLevel.bound());
}

std::string MSCodeGen::makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                         iir::Interval::Bound bound) {
  if(bound == iir::Interval::Bound::lower) {
    return makeIntervalLevelBound(dom, interval.lowerIntervalLevel());
  } else {
    return makeIntervalLevelBound(dom, interval.upperIntervalLevel());
  }
}

void MSCodeGen::generateKCacheFillStatement(MemberFunction& cudaKernel,
                                            const std::unordered_map<int, Array3i>& fieldIndexMap,
                                            const KCacheProperties& kcacheProp, int klev) const {
  std::stringstream ss;
  CodeGeneratorHelper::generateFieldAccessDeref(
      ss, ms_, stencilInstantiation_, kcacheProp.accessID_, fieldIndexMap, Array3i{0, 0, klev});
  cudaKernel.addStatement(kcacheProp.name_ + "[" + std::to_string(cacheProperties_.getKCacheIndex(
                                                       kcacheProp.accessID_, klev)) +
                          "] =" + ss.str());
}

void MSCodeGen::generatePreFillKCaches(
    MemberFunction& cudaKernel, const iir::Interval& interval,
    const std::unordered_map<int, Array3i>& fieldIndexMap) const {
  cudaKernel.addComment("Pre-fill of kcaches");

  auto intervalFields = ms_->computeFieldsAtInterval(interval);
  std::unordered_map<iir::Extents, std::vector<KCacheProperties>> kCacheProperty;

  for(const auto& cachePair : ms_->getCaches()) {
    const int accessID = cachePair.first;
    const auto& cache = cachePair.second;
    if(!CacheProperties::requiresFill(cache))
      continue;

    // for a pre-fill operation, we dont take into account the extent over the whole ms (i.e. all
    // intervals) but rather the current interval, which is the first in full vertical iteration.
    // This is because extents in the inner part of the vertical iteration can be larger, and if
    // applied at the intervals on the bounds of the iteration, they can generate out of bound
    // accesses
    auto extents = ms_->computeExtents(accessID, interval);
    if(!extents.is_initialized()) {
      continue;
    }
    auto vertExtent = (*extents)[2];

    DAWN_ASSERT(cache.getInterval().is_initialized());
    const auto cacheInterval = *(cache.getInterval());

    iir::Interval::Bound intervalBound = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                             ? iir::Interval::Bound::upper
                                             : iir::Interval::Bound::lower;

    // if the bound of a kcache is the same as the interval, indicates that we will start using the
    // kcache for the first time with this k-leg, and a pre-fill might be required
    if(cacheInterval.bound(intervalBound) == (interval.bound(intervalBound))) {
      auto cacheName = cacheProperties_.getCacheName(accessID);
      iir::Extents horizontalExtent = intervalFields.at(accessID).getExtentsRB();
      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, vertExtent);
    }
  }

  for(const auto& kcachePropPair : kCacheProperty) {
    const auto& horizontalExtent = kcachePropPair.first;
    const auto& kcachesProp = kcachePropPair.second;

    // we need to also box the fill of kcaches to avoid out-of-bounds
    cudaKernel.addBlockStatement(
        "if(iblock >= " + std::to_string(horizontalExtent[0].Minus) +
            " && iblock <= block_size_i -1 + " + std::to_string(horizontalExtent[0].Plus) +
            " && jblock >= " + std::to_string(horizontalExtent[1].Minus) +
            " && jblock <= block_size_j -1 + " + std::to_string(horizontalExtent[1].Plus) + ")",
        [&]() {
          for(const auto& kcacheProp : kcachesProp) {
            if(ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward) {
              // the last level is skipped since it will be filled in a normal kcache fill method
              for(int klev = kcacheProp.vertExtent_.Minus + 1; klev <= kcacheProp.vertExtent_.Plus;
                  ++klev) {
                generateKCacheFillStatement(cudaKernel, fieldIndexMap, kcacheProp, klev);
              }
            } else {
              for(int klev = kcacheProp.vertExtent_.Plus - 1; klev >= kcacheProp.vertExtent_.Minus;
                  --klev) {
                generateKCacheFillStatement(cudaKernel, fieldIndexMap, kcacheProp, klev);
              }
            }
          }
        });
  }
}

std::string MSCodeGen::makeLoopImpl(const iir::Extent extent, const std::string& dim,
                                    const std::string& lower, const std::string& upper,
                                    const std::string& comparison, const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "+" + std::to_string(extent.Minus) + "; " + dim +
               " " + comparison + " " + upper + "+" + std::to_string(extent.Plus) + "; " +
               increment + dim + ")")
      .str();
}

void MSCodeGen::generateFillKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                    const std::unordered_map<int, Array3i>& fieldIndexMap) const {
  cudaKernel.addComment("Center fill of kcaches");

  auto intervalFields = ms_->computeFieldsAtInterval(interval);
  std::unordered_map<iir::Extents, std::vector<KCacheProperties>> kCacheProperty;

  for(const auto& cachePair : ms_->getCaches()) {
    const int accessID = cachePair.first;
    const auto& cache = cachePair.second;
    if(!CacheProperties::requiresFill(cache))
      continue;

    DAWN_ASSERT(cache.getInterval().is_initialized());
    const auto cacheInterval = *(cache.getInterval());
    auto vertExtent = cacheProperties_.getKCacheVertExtent(accessID);

    iir::Interval::Bound intervalBound = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                             ? iir::Interval::Bound::lower
                                             : iir::Interval::Bound::upper;

    const bool cacheEndWithinInterval =
        (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
            ? interval.bound(intervalBound) >= cacheInterval.bound(intervalBound)
            : interval.bound(intervalBound) <= cacheInterval.bound(intervalBound);

    if(cacheInterval.overlaps(interval) && cacheEndWithinInterval) {
      auto cacheName = cacheProperties_.getCacheName(accessID);

      DAWN_ASSERT(intervalFields.count(accessID));
      iir::Extents horizontalExtent = intervalFields.at(accessID).getExtentsRB();

      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, vertExtent);
    }
  }

  for(const auto& kcachePropPair : kCacheProperty) {
    const auto& horizontalExtent = kcachePropPair.first;
    const auto& kcachesProp = kcachePropPair.second;

    cudaKernel.addBlockStatement(
        "if(iblock >= " + std::to_string(horizontalExtent[0].Minus) +
            " && iblock <= block_size_i -1 + " + std::to_string(horizontalExtent[0].Plus) +
            " && jblock >= " + std::to_string(horizontalExtent[1].Minus) +
            " && jblock <= block_size_j -1 + " + std::to_string(horizontalExtent[1].Plus) + ")",
        [&]() {
          for(const auto& kcacheProp : kcachesProp) {

            int offset = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                             ? kcacheProp.vertExtent_.Minus
                             : kcacheProp.vertExtent_.Plus;
            std::stringstream ss;
            CodeGeneratorHelper::generateFieldAccessDeref(ss, ms_, stencilInstantiation_,
                                                          kcacheProp.accessID_, fieldIndexMap,
                                                          Array3i{0, 0, offset});
            cudaKernel.addStatement(
                kcacheProp.name_ + "[" +
                std::to_string(cacheProperties_.getKCacheIndex(kcacheProp.accessID_, offset)) +
                "] =" + ss.str());
          }
        });
  }
}

std::string MSCodeGen::makeKLoop(const std::string dom, iir::Interval const& interval,
                                 bool kParallel) {

  iir::LoopOrderKind loopOrder = ms_->getLoopOrder();

  std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  if(kParallel) {
    lower = "kleg_lower_bound";
    upper = "kleg_upper_bound";
  }
  return (loopOrder == iir::LoopOrderKind::LK_Backward)
             ? makeLoopImpl(iir::Extent{}, "k", upper, lower, ">=", "--")
             : makeLoopImpl(iir::Extent{}, "k", lower, upper, "<=", "++");
}

bool MSCodeGen::intervalRequiresSync(const iir::Interval& interval, const iir::Stage& stage) const {
  // if the stage is the last stage, it will require a sync (to ensure we sync before the write of a
  // previous stage at the next k level), but only if the stencil is not pure vertical and ij caches
  // are used after the last sync
  int lastStageID = -1;
  // we identified the last stage that required a sync
  int lastStageIDWithSync = -1;
  for(const auto& st : ms_->getChildren()) {
    if(st->getEnclosingInterval().overlaps(interval)) {
      lastStageID = st->getStageID();
      if(st->getRequiresSync()) {
        lastStageIDWithSync = st->getStageID();
      }
    }
  }
  DAWN_ASSERT(lastStageID != -1);

  if(stage.getStageID() != lastStageID) {
    return false;
  }
  bool activateSearch = (lastStageIDWithSync == -1) ? true : false;
  for(const auto& st : ms_->getChildren()) {
    // we only activate the search to determine if IJ caches are used after last stage that was
    // sync
    if(st->getStageID() == lastStageIDWithSync) {
      activateSearch = true;
    }
    if(!activateSearch)
      continue;
    const auto& fields = st->getFields();

    // If any IJ cache is used after the last synchronized stage,
    // we will need to sync again after the last stage of the vertical loop
    for(const auto& cache : ms_->getCaches()) {
      if(cache.second.getCacheType() != iir::Cache::CacheTypeKind::IJ)
        continue;

      if(fields.count(cache.first)) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_map<iir::Extents, std::vector<MSCodeGen::KCacheProperties>>
MSCodeGen::buildKCacheProperties(const iir::Interval& interval,
                                 const iir::Cache::CacheIOPolicy policy,
                                 const bool checkStrictIntervalBound) const {

  std::unordered_map<iir::Extents, std::vector<KCacheProperties>> kCacheProperty;
  auto intervalFields = ms_->computeFieldsAtInterval(interval);

  for(const auto& IDCachePair : ms_->getCaches()) {
    const int accessID = IDCachePair.first;
    const auto& cache = IDCachePair.second;

    if(policy != cache.getCacheIOPolicy()) {
      continue;
    }
    DAWN_ASSERT(policy != iir::Cache::CacheIOPolicy::local);
    DAWN_ASSERT(cache.getInterval().is_initialized());
    const auto cacheInterval = *(cache.getInterval());
    auto vertExtent = cacheProperties_.getKCacheVertExtent(accessID);

    iir::Interval::Bound endBound = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                        ? iir::Interval::Bound::lower
                                        : iir::Interval::Bound::upper;
    const bool cacheEndWithinInterval = (interval.bound(endBound) == cacheInterval.bound(endBound));

    if(cacheInterval.contains(interval) && (!checkStrictIntervalBound || cacheEndWithinInterval)) {
      auto cacheName = cacheProperties_.getCacheName(accessID);

      DAWN_ASSERT(intervalFields.count(accessID));
      iir::Extents horizontalExtent = intervalFields.at(accessID).getExtentsRB();
      horizontalExtent[2] = {0, 0};
      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, vertExtent);
    }
  }
  return kCacheProperty;
}

void MSCodeGen::generateKCacheFlushStatement(MemberFunction& cudaKernel,
                                             const std::unordered_map<int, Array3i>& fieldIndexMap,
                                             const int accessID, std::string cacheName,
                                             const int offset) const {
  std::stringstream ss;
  CodeGeneratorHelper::generateFieldAccessDeref(ss, ms_, stencilInstantiation_, accessID,
                                                fieldIndexMap, Array3i{0, 0, offset});
  cudaKernel.addStatement(ss.str() + "= " + cacheName + "[" +
                          std::to_string(cacheProperties_.getKCacheIndex(accessID, offset)) + "]");
}

std::string MSCodeGen::kBegin(const std::string dom, iir::LoopOrderKind loopOrder,
                              iir::Interval const& interval) {

  std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  return (loopOrder == iir::LoopOrderKind::LK_Backward) ? upper : lower;
}

void MSCodeGen::generateKCacheFlushBlockStatement(
    MemberFunction& cudaKernel, const iir::Interval& interval,
    const std::unordered_map<int, Array3i>& fieldIndexMap, const KCacheProperties& kcacheProp,
    const int klev, std::string currentKLevel) const {

  const int accessID = kcacheProp.accessID_;
  const auto& cache = ms_->getCache(accessID);
  const auto& cacheInterval = *(cache.getInterval());

  int kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                             ? kcacheProp.vertExtent_.Plus
                             : kcacheProp.vertExtent_.Minus;

  // we can not flush the cache beyond the interval where the field is accessed, since that would
  // write un-initialized data back into main memory of the field. If the distance of the
  // computation interval to the interval limits of the cache is larger than the tail of the kcache
  // being flushed, we need to insert a conditional guard
  auto dist = distance(cacheInterval, interval, ms_->getLoopOrder());
  if(dist.rangeType_ != iir::IntervalDiff::RangeType::literal ||
     std::abs(dist.value) >= std::abs(kcacheTailExtent)) {
    generateKCacheFlushStatement(cudaKernel, fieldIndexMap, kcacheProp.accessID_, kcacheProp.name_,
                                 klev);
  } else {
    std::stringstream pred;
    std::string intervalKBegin = kBegin("dom", ms_->getLoopOrder(), cacheInterval);

    if(ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward) {
      pred << "if( " + intervalKBegin + " - " + currentKLevel + " >= " +
                  std::to_string(std::abs(kcacheTailExtent)) + ")";
    } else {
      pred << "if( " + currentKLevel + " - " + intervalKBegin + " >= " +
                  std::to_string(std::abs(kcacheTailExtent)) + ")";
    }
    cudaKernel.addBlockStatement(pred.str(), [&]() {
      generateKCacheFlushStatement(cudaKernel, fieldIndexMap, kcacheProp.accessID_,
                                   kcacheProp.name_, klev);
    });
  }
}

void MSCodeGen::generateFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                     const std::unordered_map<int, Array3i>& fieldIndexMap) const {
  cudaKernel.addComment("Flush of kcaches");

  auto kCacheProperty = buildKCacheProperties(interval, iir::Cache::CacheIOPolicy::flush, false);

  for(const auto& kcachePropPair : kCacheProperty) {
    const auto& horizontalExtent = kcachePropPair.first;
    const auto& kcachesProp = kcachePropPair.second;

    cudaKernel.addBlockStatement(
        "if(iblock >= " + std::to_string(horizontalExtent[0].Minus) +
            " && iblock <= block_size_i -1 + " + std::to_string(horizontalExtent[0].Plus) +
            " && jblock >= " + std::to_string(horizontalExtent[1].Minus) +
            " && jblock <= block_size_j -1 + " + std::to_string(horizontalExtent[1].Plus) + ")",
        [&]() {
          for(const auto& kcacheProp : kcachesProp) {
            int kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                       ? kcacheProp.vertExtent_.Plus
                                       : kcacheProp.vertExtent_.Minus;

            generateKCacheFlushBlockStatement(cudaKernel, interval, fieldIndexMap, kcacheProp,
                                              kcacheTailExtent, "k");
          }
        });
  }
}

void MSCodeGen::generateKCacheSlide(MemberFunction& cudaKernel,
                                    const iir::Interval& interval) const {
  cudaKernel.addComment("Slide kcaches");
  for(const auto& cachePair : ms_->getCaches()) {
    const auto& cache = cachePair.second;
    if(!cacheProperties_.isKCached(cache) ||
       ((cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::local) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::fill) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::flush) &&
        (cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::epflush)))
      continue;
    auto cacheInterval = cache.getInterval();
    DAWN_ASSERT(cacheInterval.is_initialized());
    if(!(*cacheInterval).overlaps(interval)) {
      continue;
    }

    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = cacheProperties_.getKCacheVertExtent(accessID);
    auto cacheName = cacheProperties_.getCacheName(accessID);

    for(int i = 0; i < -vertExtent.Minus + vertExtent.Plus; ++i) {
      if(ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward) {
        int maxCacheIdx = -vertExtent.Minus + vertExtent.Plus;
        cudaKernel.addStatement(cacheName + "[" + std::to_string(maxCacheIdx - i) + "] = " +
                                cacheName + "[" + std::to_string(maxCacheIdx - i - 1) + "]");
      } else {
        cudaKernel.addStatement(cacheName + "[" + std::to_string(i) + "] = " + cacheName + "[" +
                                std::to_string(i + 1) + "]");
      }
    }
  }
}

void MSCodeGen::generateFinalFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                          const std::unordered_map<int, Array3i>& fieldIndexMap,
                                          const iir::Cache::CacheIOPolicy policy) const {
  cudaKernel.addComment("Final flush of kcaches");

  DAWN_ASSERT((policy == iir::Cache::CacheIOPolicy::epflush) ||
              (policy == iir::Cache::CacheIOPolicy::flush));
  auto kCacheProperty = buildKCacheProperties(interval, policy, true);

  for(const auto& kcachePropPair : kCacheProperty) {
    const auto& horizontalExtent = kcachePropPair.first;
    const auto& kcachesProp = kcachePropPair.second;

    cudaKernel.addBlockStatement(
        "if(iblock >= " + std::to_string(horizontalExtent[0].Minus) +
            " && iblock <= block_size_i -1 + " + std::to_string(horizontalExtent[0].Plus) +
            " && jblock >= " + std::to_string(horizontalExtent[1].Minus) +
            " && jblock <= block_size_j -1 + " + std::to_string(horizontalExtent[1].Plus) + ")",
        [&]() {
          for(const auto& kcacheProp : kcachesProp) {
            const int accessID = kcacheProp.accessID_;
            const auto& cache = ms_->getCache(accessID);
            DAWN_ASSERT((cache.getInterval().is_initialized()));

            int kcacheTailExtent;
            if(policy == iir::Cache::CacheIOPolicy::flush) {
              kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                     ? kcacheProp.vertExtent_.Plus
                                     : kcacheProp.vertExtent_.Minus;
            } else if(policy == iir::Cache::CacheIOPolicy::epflush) {
              DAWN_ASSERT(cache.getWindow().is_initialized());
              auto window = *(cache.getWindow());
              kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                     ? window.m_p
                                     : window.m_m;
            } else {
              dawn_unreachable("Not valid policy for final flush");
            }

            int firstFlushLevel = kcacheTailExtent;
            iir::increment(firstFlushLevel, ms_->getLoopOrder());

            auto lastLevelComputed = ms_->lastLevelComputed(accessID);
            iir::increment(lastLevelComputed.offset_, ms_->getLoopOrder());
            auto lastKLevelStr = makeIntervalLevelBound("dom", lastLevelComputed);

            for(int klev = firstFlushLevel;
                iir::isLevelExecBeforeEqThan(klev, 0, ms_->getLoopOrder());
                iir::increment(klev, ms_->getLoopOrder())) {
              // for the final flush, we need to do an extra decrement the levels we are flushing
              // since the final flush happens after a last iterator increment beyond the inverla
              // bounds
              int klevFlushed = klev;
              iir::increment(klevFlushed, ms_->getLoopOrder(), -1);
              generateKCacheFlushBlockStatement(cudaKernel, interval, fieldIndexMap, kcacheProp,
                                                klevFlushed, lastKLevelStr);
            }
          }
        });
  }
}

void MSCodeGen::generateCudaKernelCode() {

  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(const auto& stage : iterateIIROver<iir::Stage>(*ms_)) {
    maxExtents.merge(stage->getExtents());
  }

  // fields used in the stencil
  const auto& fields = ms_->getFields();

  auto nonTempFields =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                            std::pair<int, iir::Field> const& p) {
                  return !stencilInstantiation_->isTemporaryField(p.second.getAccessID());
                }));
  // all the temp fields that are non local cache, and therefore will require the infrastructure of
  // tmp storages (allocation, iterators, etc)
  auto tempFieldsNonLocalCached =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                            std::pair<int, iir::Field> const& p) {
                  const int accessID = p.first;
                  if(!stencilInstantiation_->isTemporaryField(p.second.getAccessID()))
                    return false;
                  if(!cacheProperties_.accessIsCached(accessID))
                    return true;
                  if(ms_->getCache(accessID).getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local)
                    return false;

                  return true;
                }));

  const bool containsTemporary = !tempFieldsNonLocalCached.empty();

  std::string fnDecl = "";
  if(containsTemporary && useTmpIndex_)
    fnDecl = "template<typename TmpStorage>";
  fnDecl = fnDecl + "__global__ void";
  MemberFunction cudaKernel(fnDecl, cudaKernelName_, ss_);

  const auto& globalsMap = stencilInstantiation_->getMetaData().globalVariableMap_;
  if(!globalsMap.empty()) {
    cudaKernel.addArg("globals globals_");
  }
  cudaKernel.addArg("const int isize");
  cudaKernel.addArg("const int jsize");
  cudaKernel.addArg("const int ksize");

  std::vector<std::string> strides = CodeGeneratorHelper::generateStrideArguments(
      nonTempFields, tempFieldsNonLocalCached, stencilInstantiation_, ms_,
      CodeGeneratorHelper::FunctionArgType::FT_Callee);

  for(const auto strideArg : strides) {
    cudaKernel.addArg(strideArg);
  }

  // first we construct non temporary field arguments
  for(auto field : nonTempFields) {
    cudaKernel.addArg("gridtools::clang::float_type * const " +
                      stencilInstantiation_->getNameFromAccessID((*field).second.getAccessID()));
  }

  // then the temporary field arguments
  for(auto field : tempFieldsNonLocalCached) {
    if(useTmpIndex_) {
      cudaKernel.addArg(c_gt() + "data_view<TmpStorage>" +
                        stencilInstantiation_->getNameFromAccessID((*field).second.getAccessID()) +
                        "_dv");
    } else {
      cudaKernel.addArg("gridtools::clang::float_type * const " +
                        stencilInstantiation_->getNameFromAccessID((*field).second.getAccessID()));
    }
  }

  DAWN_ASSERT(fields.size() > 0);
  auto firstField = *(fields.begin());

  cudaKernel.startBody();
  cudaKernel.addComment("Start kernel");

  // extract raw pointers of temporaries from the data views
  if(useTmpIndex_) {
    for(auto field : tempFieldsNonLocalCached) {
      std::string fieldName =
          stencilInstantiation_->getNameFromAccessID((*field).second.getAccessID());

      cudaKernel.addStatement("gridtools::clang::float_type* " + fieldName + " = &" + fieldName +
                              "_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0)");
    }
  }

  generateIJCacheDecl(cudaKernel);
  generateKCacheDecl(cudaKernel);

  unsigned int ntx = blockSize_[0];
  unsigned int nty = blockSize_[1];
  cudaKernel.addStatement("const unsigned int nx = isize");
  cudaKernel.addStatement("const unsigned int ny = jsize");
  cudaKernel.addStatement("const int block_size_i = (blockIdx.x + 1) * " + std::to_string(ntx) +
                          " < nx ? " + std::to_string(ntx) + " : nx - blockIdx.x * " +
                          std::to_string(ntx));
  cudaKernel.addStatement("const int block_size_j = (blockIdx.y + 1) * " + std::to_string(nty) +
                          " < ny ? " + std::to_string(nty) + " : ny - blockIdx.y * " +
                          std::to_string(nty));

  cudaKernel.addComment("computing the global position in the physical domain");
  cudaKernel.addComment("In a typical cuda block we have the following regions");
  cudaKernel.addComment("aa bbbbbbbb cc");
  cudaKernel.addComment("aa bbbbbbbb cc");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("hh dddddddd ii");
  cudaKernel.addComment("ee ffffffff gg");
  cudaKernel.addComment("ee ffffffff gg");
  cudaKernel.addComment("Regions b,d,f have warp (or multiple of warp size)");
  cudaKernel.addComment("Size of regions a, c, h, i, e, g are determined by max_extent_t");
  cudaKernel.addComment(
      "Regions b,d,f are easily executed by dedicated warps (one warp for each line)");
  cudaKernel.addComment("Regions (a,h,e) and (c,i,g) are executed by two specialized warp");

  // jboundary_limit determines the number of warps required to execute (b,d,f)");
  int jboundary_limit = (int)nty + +maxExtents[1].Plus - maxExtents[1].Minus;
  int iminus_limit = jboundary_limit + (maxExtents[0].Minus < 0 ? 1 : 0);
  int iplus_limit = iminus_limit + (maxExtents[0].Plus > 0 ? 1 : 0);

  cudaKernel.addStatement("int iblock = " + std::to_string(maxExtents[0].Minus) + " - 1");
  cudaKernel.addStatement("int jblock = " + std::to_string(maxExtents[1].Minus) + " - 1");
  cudaKernel.addBlockStatement("if(threadIdx.y < +" + std::to_string(jboundary_limit) + ")", [&]() {
    cudaKernel.addStatement("iblock = threadIdx.x");
    cudaKernel.addStatement("jblock = (int)threadIdx.y + " + std::to_string(maxExtents[1].Minus));
  });
  if(maxExtents[0].Minus < 0) {
    cudaKernel.addBlockStatement(
        "else if(threadIdx.y < +" + std::to_string(iminus_limit) + ")", [&]() {
          int paddedBoundary_ = paddedBoundary(maxExtents[0].Minus);

          // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough
          //  threads
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize_[0]),
                          "not enought cuda threads");

          cudaKernel.addStatement("iblock = -" + std::to_string(paddedBoundary_) +
                                  " + (int)threadIdx.x % " + std::to_string(paddedBoundary_));
          cudaKernel.addStatement("jblock = (int)threadIdx.x / " + std::to_string(paddedBoundary_) +
                                  "+" + std::to_string(maxExtents[1].Minus));
        });
  }
  if(maxExtents[0].Plus > 0) {
    cudaKernel.addBlockStatement(
        "else if(threadIdx.y < " + std::to_string(iplus_limit) + ")", [&]() {
          int paddedBoundary_ = paddedBoundary(maxExtents[0].Plus);
          // we dedicate one warp to execute regions (c,i,g), so here we make sure we have enough
          //    threads
          // we dedicate one warp to execute regions (a,h,e), so here we make sure we have enough
          //  threads
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize_[0]),
                          "not enought cuda threads");

          cudaKernel.addStatement("iblock = threadIdx.x % " + std::to_string(paddedBoundary_) +
                                  " + " + std::to_string(ntx));
          cudaKernel.addStatement("jblock = (int)threadIdx.x / " + std::to_string(paddedBoundary_) +
                                  "+" + std::to_string(maxExtents[1].Minus));
        });
  }

  std::unordered_map<int, Array3i> fieldIndexMap;
  std::unordered_map<std::string, Array3i> indexIterators;

  for(auto field : nonTempFields) {
    Array3i dims{-1, -1, -1};
    for(const auto& fieldInfo : ms_->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }
    DAWN_ASSERT(std::accumulate(dims.begin(), dims.end(), 0) != -3);
    fieldIndexMap.emplace((*field).second.getAccessID(), dims);
    indexIterators.emplace(CodeGeneratorHelper::indexIteratorName(dims), dims);
  }
  for(auto field : tempFieldsNonLocalCached) {
    Array3i dims{1, 1, 1};
    fieldIndexMap.emplace((*field).second.getAccessID(), dims);
    indexIterators.emplace(CodeGeneratorHelper::indexIteratorName(dims), dims);
  }

  cudaKernel.addComment("initialized iterators");
  for(auto index : indexIterators) {
    std::string idxStmt = "int idx" + index.first + " = ";
    bool init = false;
    if(index.second[0] != 1 && index.second[1] != 1) {
      idxStmt = idxStmt + "0";
    }
    if(index.second[0]) {
      init = true;
      idxStmt = idxStmt + "(blockIdx.x*" + std::to_string(ntx) + "+iblock)*1";
    }
    if(index.second[1]) {
      if(init) {
        idxStmt = idxStmt + "+";
      }
      idxStmt = idxStmt + "(blockIdx.y*" + std::to_string(nty) + "+jblock)*" +
                CodeGeneratorHelper::generateStrideName(1, index.second);
    }
    cudaKernel.addStatement(idxStmt);
  }

  if(cacheProperties_.hasIJCaches()) {
    generateIJCacheIndexInit(cudaKernel);
  }

  if(containsTemporary) {
    generateTmpIndexInit(cudaKernel);
  }

  // compute the partition of the intervals
  auto partitionIntervals = CodeGeneratorHelper::computePartitionOfIntervals(ms_);

  DAWN_ASSERT(!partitionIntervals.empty());

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation_, fieldIndexMap, ms_, cacheProperties_,
                                       blockSize_);

  iir::Interval::IntervalLevel lastKCell{0, 0};
  lastKCell = advance(lastKCell, ms_->getLoopOrder(), -1);

  bool firstInterval = true;
  int klegDeclared = false;
  for(auto interval : partitionIntervals) {

    // If execution is parallel we want to place the interval in a forward order
    if((solveKLoopInParallel_) && (interval.lowerBound() > interval.upperBound())) {
      interval.invert();
    }
    iir::IntervalDiff kmin{iir::IntervalDiff::RangeType::literal, 0};
    // if there is a jump between the last level of previous interval and the first level of this
    // interval, we advance the iterators

    iir::Interval::IntervalLevel nextLevel =
        computeNextLevelToProcess(interval, ms_->getLoopOrder());
    auto jump = distance(lastKCell, nextLevel);
    if((std::abs(jump.value) != 1) || (jump.rangeType_ != iir::IntervalDiff::RangeType::literal)) {
      auto lastKCellp1 = advance(lastKCell, ms_->getLoopOrder(), 1);
      kmin = distance(lastKCellp1, nextLevel);

      for(auto index : indexIterators) {
        if(index.second[2] && !kmin.null() && !((solveKLoopInParallel_) && firstInterval)) {
          cudaKernel.addComment("jump iterators to match the beginning of next interval");
          cudaKernel.addStatement("idx" + index.first + " += " +
                                  CodeGeneratorHelper::generateStrideName(2, index.second) + "*(" +
                                  intervalDiffToString(kmin, "ksize - 1") + ")");
        }
      }
      if(useTmpIndex_ && !kmin.null() && !((solveKLoopInParallel_) && firstInterval)) {
        cudaKernel.addComment("jump tmp iterators to match the beginning of next interval");
        cudaKernel.addStatement("idx_tmp += kstride_tmp*(" +
                                intervalDiffToString(kmin, "ksize - 1") + ")");
      }
    }

    if(solveKLoopInParallel_) {
      // advance the iterators to the first k index of the block or the first position of the
      // interval
      for(auto index : indexIterators) {
        if(index.second[2]) {
          // the advance should correspond to max(beginning of the parallel block, beginning
          // interval)
          // but only for the first interval we force the advance to the beginning of the parallel
          // block
          std::string step = intervalDiffToString(kmin, "ksize - 1");
          if(firstInterval) {
            step = "max(" + step + "," + " blockIdx.z * " + std::to_string(blockSize_[2]) + ") * " +
                   CodeGeneratorHelper::generateStrideName(2, index.second);

            cudaKernel.addComment("jump iterators to match the intersection of beginning of next "
                                  "interval and the parallel execution block ");
            cudaKernel.addStatement("idx" + index.first + " += " + step);
          }
        }
      }
      if(useTmpIndex_) {
        cudaKernel.addComment("jump tmp iterators to match the intersection of beginning of next "
                              "interval and the parallel execution block ");
        cudaKernel.addStatement("idx_tmp += max(" + intervalDiffToString(kmin, "ksize - 1") +
                                ", kstride_tmp * blockIdx.z * " + std::to_string(blockSize_[2]) +
                                ")");
      }
    }

    if(solveKLoopInParallel_) {
      // define the loop bounds of each parallel kleg
      std::string lower = makeIntervalBound("dom", interval, iir::Interval::Bound::lower);
      std::string upper = makeIntervalBound("dom", interval, iir::Interval::Bound::upper);
      cudaKernel.addStatement((!klegDeclared ? std::string("int ") : std::string("")) +
                              "kleg_lower_bound = max(" + lower + ",blockIdx.z*" +
                              std::to_string(blockSize_[2]) + ")");
      cudaKernel.addStatement((!klegDeclared ? std::string("int ") : std::string("")) +
                              "kleg_upper_bound = min(" + upper + ",(blockIdx.z+1)*" +
                              std::to_string(blockSize_[2]) + "-1);");
      klegDeclared = true;
    }

    if(!solveKLoopInParallel_) {
      generatePreFillKCaches(cudaKernel, interval, fieldIndexMap);
    }

    // for each interval, we generate naive nested loops
    cudaKernel.addBlockStatement(makeKLoop("dom", interval, solveKLoopInParallel_), [&]() {

      if(!solveKLoopInParallel_) {
        generateFillKCaches(cudaKernel, interval, fieldIndexMap);
      }

      for(const auto& stagePtr : ms_->getChildren()) {
        const iir::Stage& stage = *stagePtr;
        const auto& extent = stage.getExtents();
        iir::MultiInterval enclosingInterval;

        // TODO add the enclosing interval in derived ?
        for(const auto& doMethodPtr : stage.getChildren()) {
          enclosingInterval.insert(doMethodPtr->getInterval());
        }
        if(!enclosingInterval.overlaps(interval))
          continue;

        // only add sync if there are data dependencies
        if(stage.getRequiresSync()) {
          cudaKernel.addStatement("__syncthreads()");
        }

        cudaKernel.addBlockStatement(
            "if(iblock >= " + std::to_string(extent[0].Minus) + " && iblock <= block_size_i -1 + " +
                std::to_string(extent[0].Plus) + " && jblock >= " +
                std::to_string(extent[1].Minus) + " && jblock <= block_size_j -1 + " +
                std::to_string(extent[1].Plus) + ")",
            [&]() {
              // Generate Do-Method
              for(const auto& doMethodPtr : stage.getChildren()) {
                const iir::DoMethod& doMethod = *doMethodPtr;
                if(!doMethod.getInterval().overlaps(interval))
                  continue;
                for(const auto& statementAccessesPair : doMethod.getChildren()) {
                  statementAccessesPair->getStatement()->ASTStmt->accept(stencilBodyCXXVisitor);
                  cudaKernel << stencilBodyCXXVisitor.getCodeAndResetStream();
                }
              }
            });
        // only add sync if there are data dependencies
        if(intervalRequiresSync(interval, stage)) {
          cudaKernel.addStatement("__syncthreads()");
        }
      }

      if(!solveKLoopInParallel_) {
        generateFlushKCaches(cudaKernel, interval, fieldIndexMap);
      }

      generateKCacheSlide(cudaKernel, interval);
      cudaKernel.addComment("increment iterators");
      std::string incStr = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward) ? "-=" : "+=";

      for(auto index : indexIterators) {
        if(index.second[2]) {
          cudaKernel.addStatement("idx" + index.first + incStr +
                                  CodeGeneratorHelper::generateStrideName(2, index.second));
        }
      }
      if(useTmpIndex_) {
        cudaKernel.addStatement("idx_tmp " + incStr + " kstride_tmp");
      }
    });
    if(!solveKLoopInParallel_) {
      generateFinalFlushKCaches(cudaKernel, interval, fieldIndexMap,
                                iir::Cache::CacheIOPolicy::flush);
      generateFinalFlushKCaches(cudaKernel, interval, fieldIndexMap,
                                iir::Cache::CacheIOPolicy::epflush);
    }

    lastKCell = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                    ? interval.lowerIntervalLevel()
                    : interval.upperIntervalLevel();
    firstInterval = false;
  }

  cudaKernel.commit();
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
