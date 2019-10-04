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

#include "dawn/CodeGen/Cuda/MSCodeGen.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGen.h"
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/Support/ContainerUtils.h"
#include "dawn/Support/IndexRange.h"
#include <functional>
#include <numeric>

namespace dawn {
namespace codegen {
namespace cuda {
MSCodeGen::MSCodeGen(std::stringstream& ss, const std::unique_ptr<iir::MultiStage>& ms,
                     const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                     const CacheProperties& cacheProperties,
                     CudaCodeGen::CudaCodeGenOptions options)
    : ss_(ss), ms_(ms), stencilInstantiation_(stencilInstantiation),
      metadata_(stencilInstantiation->getMetaData()), cacheProperties_(cacheProperties),
      useCodeGenTemporaries_(CodeGeneratorHelper::useTemporaries(
                                 ms->getParent(), stencilInstantiation->getMetaData()) &&
                             ms->hasMemAccessTemporaries()),
      cudaKernelName_(CodeGeneratorHelper::buildCudaKernelName(stencilInstantiation_, ms_)),
      blockSize_(stencilInstantiation_->getIIR()->getBlockSize()),
      solveKLoopInParallel_(CodeGeneratorHelper::solveKLoopInParallel(ms_)), options_(options) {}

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

    if(cache.getCacheType() != iir::Cache::CacheTypeKind::K)
      continue;

    if(cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::local && solveKLoopInParallel_)
      continue;
    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = ms_->getKCacheVertExtent(accessID);

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
        "= iblock + " + std::to_string(cacheProperties_.getOffsetCommonIJCache(0)) +
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

  if(!useCodeGenTemporaries_)
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
  CodeGeneratorHelper::generateFieldAccessDeref(ss, ms_, stencilInstantiation_->getMetaData(),
                                                kcacheProp.accessID_, fieldIndexMap,
                                                Array3i{0, 0, klev});
  cudaKernel.addStatement(
      kcacheProp.name_ + "[" +
      std::to_string(cacheProperties_.getKCacheIndex(kcacheProp.accessID_, klev)) +
      "] =" + ss.str());
}

iir::MultiInterval
MSCodeGen::intervalNotPreviouslyAccessed(const int accessID, const iir::Interval& targetInterval,
                                         const iir::Interval& queryInterval) const {
  iir::MultiInterval res{targetInterval};
  auto partitionIntervals = CodeGeneratorHelper::computePartitionOfIntervals(ms_);

  for(const auto& aInterval : partitionIntervals) {
    // we only need to check intervals that were computed before the current interval of execution
    if(aInterval.overlaps(queryInterval)) {
      break;
    }
    for(auto const& doMethod : iterateIIROver<iir::DoMethod>(*ms_)) {
      if(!doMethod->getInterval().overlaps(aInterval)) {
        continue;
      }
      if(!doMethod->hasField(accessID)) {
        continue;
      }
      auto doMethodInterval = doMethod->getInterval().intersect(aInterval);
      const auto& field = doMethod->getField(accessID);
      auto accessedInterval = doMethodInterval.extendInterval(field.getExtents()[2]);

      if(accessedInterval.overlaps(targetInterval)) {
        res.substract(accessedInterval);
      }
    }
  }
  return res;
}

void MSCodeGen::generatePreFillKCaches(
    MemberFunction& cudaKernel, const iir::Interval& interval,
    const std::unordered_map<int, Array3i>& fieldIndexMap) const {
  cudaKernel.addComment("Pre-fill of kcaches");

  // the algorithm consists of two parts:
  // 1) identify all the caches that require a prefill for each of the intervals of the partition.
  //    Insert into kCacheProperty
  // 2) generation of the prefill code for all cache/intervals in KCacheProperty
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
    if(!extents) {
      continue;
    }
    auto intervalVertExtent = (*extents)[2];

    DAWN_ASSERT(cache.getInterval());

    // if only one level is accessed (the head of the cache) this level will be provided by the fill
    // operation
    if(intervalVertExtent.Minus == intervalVertExtent.Plus)
      continue;

    // check the interval of levels accessed beyond the iteration interval. This will mark all the
    // levels of the kcache that will be accessed but are not filled (they will have to be
    // prefilled) at the beginning of the processing of the interval
    auto outOfRangeAccessedInterval =
        (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
            ? interval.crop(iir::Interval::Bound::upper,
                            // the last level of Minus if not required since will be filled by the
                            // head fill method
                            {intervalVertExtent.Minus + 1, intervalVertExtent.Plus})
            : interval.crop(iir::Interval::Bound::lower,
                            // the last level of Plus if not required since will be filled by the
                            // head fill method
                            {intervalVertExtent.Minus, intervalVertExtent.Plus - 1});

    /// we check if the levels beyond the iteration interval are filled already by the processing of
    /// previous intervals
    auto notYetAccessedInterval =
        intervalNotPreviouslyAccessed(accessID, outOfRangeAccessedInterval, interval);
    if(!notYetAccessedInterval.empty()) {
      // if the outOfRangeAccessInterval has not been accessed (and therefore filled in the cache)
      // by the processing of a previous interval we need to add a prefill action
      auto cacheName = cacheProperties_.getCacheName(accessID);
      iir::Extents horizontalExtent = intervalFields.at(accessID).getExtentsRB();
      auto firstInterval = notYetAccessedInterval.getIntervals()[0];
      auto lastInterval =
          notYetAccessedInterval.getIntervals()[notYetAccessedInterval.numPartitions() - 1];

      iir::Interval preFillInterval(firstInterval.lowerLevel(), lastInterval.upperLevel(),
                                    firstInterval.lowerOffset(), lastInterval.upperOffset());

      auto preFillMarkLevel = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                  ? interval.upperIntervalLevel()
                                  : interval.lowerIntervalLevel();

      iir::Extent preFillExtent{
          iir::distance(preFillMarkLevel, preFillInterval.lowerIntervalLevel()).value,
          iir::distance(preFillMarkLevel, preFillInterval.upperIntervalLevel()).value};

      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, preFillExtent);
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
              for(int klev = kcacheProp.intervalVertExtent_.Minus;
                  klev <= kcacheProp.intervalVertExtent_.Plus; ++klev) {
                generateKCacheFillStatement(cudaKernel, fieldIndexMap, kcacheProp, klev);
              }
            } else {
              for(int klev = kcacheProp.intervalVertExtent_.Plus;
                  klev >= kcacheProp.intervalVertExtent_.Minus; --klev) {
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
  cudaKernel.addComment("Head fill of kcaches");

  auto intervalFields = ms_->computeFieldsAtInterval(interval);
  std::unordered_map<iir::Extents, std::vector<KCacheProperties>> kCacheProperty;

  for(const auto& cachePair : ms_->getCaches()) {
    const int accessID = cachePair.first;
    const auto& cache = cachePair.second;
    if(!CacheProperties::requiresFill(cache))
      continue;

    DAWN_ASSERT(cache.getInterval());
    const auto cacheInterval = *(cache.getInterval());
    auto extents = ms_->computeExtents(accessID, interval);
    if(!extents) {
      continue;
    }

    auto intervalVertExtent = (*extents)[2];

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

      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, intervalVertExtent);
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
                             ? kcacheProp.intervalVertExtent_.Minus
                             : kcacheProp.intervalVertExtent_.Plus;
            std::stringstream ss;
            CodeGeneratorHelper::generateFieldAccessDeref(
                ss, ms_, stencilInstantiation_->getMetaData(), kcacheProp.accessID_, fieldIndexMap,
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

bool MSCodeGen::checkIfCacheNeedsToFlush(const iir::Cache& cache, iir::Interval interval) const {
  DAWN_ASSERT(cache.getInterval());

  const iir::Interval& cacheInterval = *(cache.getInterval());
  if(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::epflush) {
    auto epflushWindowInterval = cache.getWindowInterval(
        (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Forward) ? iir::Interval::Bound::upper
                                                                : iir::Interval::Bound::lower);
    return epflushWindowInterval.overlaps(interval);
  } else {
    return cacheInterval.contains(interval);
  }
}

std::unordered_map<iir::Extents, std::vector<MSCodeGen::KCacheProperties>>
MSCodeGen::buildKCacheProperties(const iir::Interval& interval,
                                 const iir::Cache::CacheIOPolicy policy) const {

  std::unordered_map<iir::Extents, std::vector<KCacheProperties>> kCacheProperty;
  auto intervalFields = ms_->computeFieldsAtInterval(interval);

  for(const auto& IDCachePair : ms_->getCaches()) {
    const int accessID = IDCachePair.first;
    const auto& cache = IDCachePair.second;

    if(policy != cache.getCacheIOPolicy()) {
      continue;
    }
    DAWN_ASSERT(policy != iir::Cache::CacheIOPolicy::local);
    DAWN_ASSERT(cache.getInterval());
    auto extents = ms_->computeExtents(accessID, interval);
    if(!extents) {
      continue;
    }
    auto applyEpflush = checkIfCacheNeedsToFlush(cache, interval);

    if(applyEpflush) {
      auto cacheName = cacheProperties_.getCacheName(accessID);
      auto intervalVertExtent = (*extents)[2];

      DAWN_ASSERT(intervalFields.count(accessID));
      iir::Extents horizontalExtent = intervalFields.at(accessID).getExtentsRB();
      horizontalExtent[2] = {0, 0};
      kCacheProperty[horizontalExtent].emplace_back(cacheName, accessID, intervalVertExtent);
    }
  }
  return kCacheProperty;
}

void MSCodeGen::generateKCacheFlushStatement(MemberFunction& cudaKernel,
                                             const std::unordered_map<int, Array3i>& fieldIndexMap,
                                             const int accessID, std::string cacheName,
                                             const int offset) const {
  std::stringstream ss;
  CodeGeneratorHelper::generateFieldAccessDeref(ss, ms_, stencilInstantiation_->getMetaData(),
                                                accessID, fieldIndexMap, Array3i{0, 0, offset});
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
                             ? kcacheProp.intervalVertExtent_.Plus
                             : kcacheProp.intervalVertExtent_.Minus;

  // we can not flush the cache beyond the interval where the field is accessed, since that would
  // write un-initialized data back into main memory of the field. If the distance of the
  // computation interval to the interval limits of the cache is larger than the tail of the
  // kcache
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
      pred << "if( " + intervalKBegin + " - " + currentKLevel +
                  " >= " + std::to_string(std::abs(kcacheTailExtent)) + ")";
    } else {
      pred << "if( " + currentKLevel + " - " + intervalKBegin +
                  " >= " + std::to_string(std::abs(kcacheTailExtent)) + ")";
    }
    cudaKernel.addBlockStatement(pred.str(), [&]() {
      generateKCacheFlushStatement(cudaKernel, fieldIndexMap, kcacheProp.accessID_,
                                   kcacheProp.name_, klev);
    });
  }
}

void MSCodeGen::generateFlushKCaches(MemberFunction& cudaKernel, const iir::Interval& interval,
                                     const std::unordered_map<int, Array3i>& fieldIndexMap,
                                     iir::Cache::CacheIOPolicy policy) const {
  cudaKernel.addComment("Flush of kcaches");

  auto kCacheProperty = buildKCacheProperties(interval, policy);

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
            // we flush the last level of the cache, that is determined by its size
            int kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                       ? kcacheProp.intervalVertExtent_.Plus
                                       : kcacheProp.intervalVertExtent_.Minus;

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
    if(!cacheProperties_.isKCached(cache))
      continue;
    auto cacheInterval = cache.getInterval();
    DAWN_ASSERT(cacheInterval);
    if(!(*cacheInterval).overlaps(interval)) {
      continue;
    }

    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = ms_->getKCacheVertExtent(accessID);
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
              (policy == iir::Cache::CacheIOPolicy::flush) ||
              (policy == iir::Cache::CacheIOPolicy::fill_and_flush));
  auto kCacheProperty = buildKCacheProperties(interval, policy);

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
            DAWN_ASSERT((cache.getInterval()));

            int kcacheTailExtent;
            if((policy == iir::Cache::CacheIOPolicy::flush) ||
               (policy == iir::Cache::CacheIOPolicy::fill_and_flush)) {
              kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                     ? kcacheProp.intervalVertExtent_.Plus
                                     : kcacheProp.intervalVertExtent_.Minus;
            } else if(policy == iir::Cache::CacheIOPolicy::epflush) {
              DAWN_ASSERT(cache.getWindow());
              auto intervalToFlush =
                  cache
                      .getWindowInterval((ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                             ? iir::Interval::Bound::lower
                                             : iir::Interval::Bound::upper)
                      .intersect(interval);
              auto distance_ = iir::distance(intervalToFlush.lowerIntervalLevel(),
                                             intervalToFlush.upperIntervalLevel()) +
                               1;
              DAWN_ASSERT(distance_.rangeType_ == iir::IntervalDiff::RangeType::literal);

              kcacheTailExtent = (ms_->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                                     ? distance_.value
                                     : -distance_.value;
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
  const auto fields = support::orderMap(ms_->getFields());

  auto nonTempFields = makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>(
                                             [&](std::pair<int, iir::Field> const& p) {
                                               return !metadata_.isAccessType(
                                                   iir::FieldAccessType::FAT_StencilTemporary,
                                                   p.second.getAccessID());
                                             }));
  // all the temp fields that are non local cache, and therefore will require the infrastructure
  // of
  // tmp storages (allocation, iterators, etc)
  auto tempFieldsNonLocalCached =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>(
                            [&](std::pair<int, iir::Field> const& p) {
                              const int accessID = p.first;
                              return ms_->isMemAccessTemporary(accessID);
                            }));

  std::string fnDecl = "";
  if(useCodeGenTemporaries_)
    fnDecl = "template<typename TmpStorage>";
  fnDecl = fnDecl + "__global__ void";

  int maxThreadsPerBlock =
      blockSize_[0] * (blockSize_[1] + maxExtents[1].Plus - maxExtents[1].Minus +
                       (maxExtents[0].Minus < 0 ? 1 : 0) + (maxExtents[0].Plus > 0 ? 1 : 0));

  int nSM = options_.nsms;
  int maxBlocksPerSM = options_.maxBlocksPerSM;

  std::string domain_size = options_.domainSize;
  if(nSM > 0 && !domain_size.empty()) {
    if(maxBlocksPerSM <= 0) {
      throw std::runtime_error("--max-blocks-sm must be defined");
    }
    std::istringstream idomain_size(domain_size);
    std::string arg;
    getline(idomain_size, arg, ',');
    int isize = std::stoi(arg);
    getline(idomain_size, arg, ',');
    int jsize = std::stoi(arg);

    int minBlocksPerSM = isize * jsize / (blockSize_[0] * blockSize_[1]);
    if(solveKLoopInParallel_)
      minBlocksPerSM *= 80 / blockSize_[2];
    minBlocksPerSM /= nSM;

    fnDecl = fnDecl + " __launch_bounds__(" + std::to_string(maxThreadsPerBlock) + "," +
             std::to_string(std::min(maxBlocksPerSM, minBlocksPerSM)) + ") ";
  } else {
    fnDecl = fnDecl + " __launch_bounds__(" + std::to_string(maxThreadsPerBlock) + ") ";
  }
  MemberFunction cudaKernel(fnDecl, cudaKernelName_, ss_);

  const auto& globalsMap = stencilInstantiation_->getIIR()->getGlobalVariableMap();
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
  for(const auto& fieldPair : nonTempFields) {
    cudaKernel.addArg("gridtools::clang::float_type * const " +
                      metadata_.getFieldNameFromAccessID(fieldPair.second.getAccessID()));
  }

  // then the temporary field arguments
  for(const auto& fieldPair : tempFieldsNonLocalCached) {
    if(useCodeGenTemporaries_) {
      cudaKernel.addArg(c_gt() + "data_view<TmpStorage>" +
                        metadata_.getFieldNameFromAccessID(fieldPair.second.getAccessID()) + "_dv");
    } else {
      cudaKernel.addArg("gridtools::clang::float_type * const " +
                        metadata_.getFieldNameFromAccessID(fieldPair.second.getAccessID()));
    }
  }

  DAWN_ASSERT(fields.size() > 0);

  cudaKernel.startBody();
  cudaKernel.addComment("Start kernel");

  // extract raw pointers of temporaries from the data views
  if(useCodeGenTemporaries_) {
    for(const auto& fieldPair : tempFieldsNonLocalCached) {
      std::string fieldName = metadata_.getFieldNameFromAccessID(fieldPair.second.getAccessID());

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

  for(const auto& fieldPair : nonTempFields) {
    Array3i dims{-1, -1, -1};
    for(const auto& fieldInfo : ms_->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == fieldPair.second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }
    DAWN_ASSERT(std::accumulate(dims.begin(), dims.end(), 0) != -3);
    fieldIndexMap.emplace(fieldPair.second.getAccessID(), dims);
    indexIterators.emplace(CodeGeneratorHelper::indexIteratorName(dims), dims);
  }
  for(const auto& fieldPair : tempFieldsNonLocalCached) {
    Array3i dims{1, 1, 1};
    fieldIndexMap.emplace(fieldPair.second.getAccessID(), dims);
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

  generateTmpIndexInit(cudaKernel);

  // compute the partition of the intervals
  auto partitionIntervals = CodeGeneratorHelper::computePartitionOfIntervals(ms_);

  DAWN_ASSERT(!partitionIntervals.empty());

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation_->getMetaData(), fieldIndexMap, ms_,
                                       cacheProperties_, blockSize_);

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
      if(useCodeGenTemporaries_ && !kmin.null() && !((solveKLoopInParallel_) && firstInterval)) {
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
      if(useCodeGenTemporaries_) {
        cudaKernel.addComment("jump tmp iterators to match the intersection of beginning of next "
                              "interval and the parallel execution block ");
        cudaKernel.addStatement("idx_tmp += max(" + intervalDiffToString(kmin, "ksize - 1") +
                                ", blockIdx.z * " + std::to_string(blockSize_[2]) +
                                ") * kstride_tmp");
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
                std::to_string(extent[0].Plus) +
                " && jblock >= " + std::to_string(extent[1].Minus) +
                " && jblock <= block_size_j -1 + " + std::to_string(extent[1].Plus) + ")",
            [&]() {
              // Generate Do-Method
              for(const auto& doMethodPtr : stage.getChildren()) {
                const iir::DoMethod& doMethod = *doMethodPtr;
                if(!doMethod.getInterval().overlaps(interval))
                  continue;
                for(const auto& statementAccessesPair : doMethod.getChildren()) {
                  statementAccessesPair->getStatement()->accept(stencilBodyCXXVisitor);
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
        generateFlushKCaches(cudaKernel, interval, fieldIndexMap, iir::Cache::CacheIOPolicy::flush);
        generateFlushKCaches(cudaKernel, interval, fieldIndexMap,
                             iir::Cache::CacheIOPolicy::fill_and_flush);
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
      if(useCodeGenTemporaries_) {
        cudaKernel.addStatement("idx_tmp " + incStr + " kstride_tmp");
      }
    });
    if(!solveKLoopInParallel_) {
      generateFinalFlushKCaches(cudaKernel, interval, fieldIndexMap,
                                iir::Cache::CacheIOPolicy::fill_and_flush);
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
