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

#include "dawn/CodeGen/Cuda/CudaCodeGen.h"
#include "dawn/CodeGen/CXXUtil.h"
#include "dawn/CodeGen/CodeGenProperties.h"
#include "dawn/CodeGen/Cuda/ASTStencilBody.h"
#include "dawn/CodeGen/Cuda/ASTStencilDesc.h"
#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/Optimizer/OptimizerContext.h"
#include "dawn/Optimizer/PassInlining.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Assert.h"
#include "dawn/Support/Logging.h"
#include "dawn/Support/StringUtil.h"
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace dawn {
namespace codegen {
namespace cuda {

static std::string makeLoopImpl(const iir::Extent extent, const std::string& dim,
                                const std::string& lower, const std::string& upper,
                                const std::string& comparison, const std::string& increment) {
  return Twine("for(int " + dim + " = " + lower + "+" + std::to_string(extent.Minus) + "; " + dim +
               " " + comparison + " " + upper + "+" + std::to_string(extent.Plus) + "; " +
               increment + dim + ")")
      .str();
}

static std::string makeIntervalBound(const std::string dom, iir::Interval const& interval,
                                     iir::Interval::Bound bound) {
  return interval.levelIsEnd(bound) ? " ksize - 1 + " + std::to_string(interval.offset(bound))
                                    : std::to_string(interval.bound(bound));
}

static std::string makeKLoop(const std::string dom, const std::array<unsigned int, 3> blockSize,
                             iir::LoopOrderKind loopOrder, iir::Interval const& interval,
                             bool kParallel) {

  std::string lower = makeIntervalBound(dom, interval, iir::Interval::Bound::lower);
  std::string upper = makeIntervalBound(dom, interval, iir::Interval::Bound::upper);

  if(kParallel) {
    lower = "max(" + lower + ",blockIdx.z*" + std::to_string(blockSize[2]) + ")";
    upper = "min(" + upper + ",(blockIdx.z+1)*" + std::to_string(blockSize[2]) + "-1)";
  }
  return (loopOrder == iir::LoopOrderKind::LK_Backward)
             ? makeLoopImpl(iir::Extent{}, "k", upper, lower, ">=", "--")
             : makeLoopImpl(iir::Extent{}, "k", lower, upper, "<=", "++");
}

CudaCodeGen::CudaCodeGen(OptimizerContext* context) : CodeGen(context) {}

CudaCodeGen::~CudaCodeGen() {}

std::string
CudaCodeGen::buildCudaKernelName(const std::shared_ptr<iir::StencilInstantiation>& instantiation,
                                 const std::unique_ptr<iir::MultiStage>& ms) {
  return instantiation->getName() + "_stencil" + std::to_string(ms->getParent()->getStencilID()) +
         "_ms" + std::to_string(ms->getID()) + "_kernel";
}

void CudaCodeGen::generateIJCacheDecl(
    MemberFunction& kernel, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const iir::MultiStage& ms, const CacheProperties& cacheProperties, Array3ui blockSize) const {
  for(const auto& cacheP : ms.getCaches()) {
    const iir::Cache& cache = cacheP.second;
    if(cache.getCacheType() != iir::Cache::CacheTypeKind::IJ)
      continue;
    DAWN_ASSERT(cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local);

    const int accessID = cache.getCachedFieldAccessID();
    const auto& maxExtents = cacheProperties.getCacheExtent(accessID);

    kernel.addStatement(
        "__shared__ gridtools::clang::float_type " + cacheProperties.getCacheName(accessID) + "[" +
        std::to_string(blockSize[0] + (maxExtents[0].Plus - maxExtents[0].Minus)) + "*" +
        std::to_string(blockSize[1] + (maxExtents[1].Plus - maxExtents[1].Minus)) + "]");
  }
}

void CudaCodeGen::generateKCacheDecl(MemberFunction& kernel, const iir::MultiStage& ms,
                                     const CacheProperties& cacheProperties) const {
  for(const auto& cacheP : ms.getCaches()) {
    const iir::Cache& cache = cacheP.second;
    if(cache.getCacheType() != iir::Cache::CacheTypeKind::K ||
       cache.getCacheIOPolicy() != iir::Cache::CacheIOPolicy::local)
      continue;

    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = cacheProperties.getKCacheVertExtent(accessID);

    kernel.addStatement("gridtools::clang::float_type " + cacheProperties.getCacheName(accessID) +
                        "[" + std::to_string(-vertExtent.Minus + vertExtent.Plus + 1) + "]");
  }
}

std::vector<iir::Interval>
CudaCodeGen::computePartitionOfIntervals(const std::unique_ptr<iir::MultiStage>& ms) const {
  auto intervals_set = ms->getIntervals();
  std::vector<iir::Interval> intervals_v;
  std::copy(intervals_set.begin(), intervals_set.end(), std::back_inserter(intervals_v));

  // compute the partition of the intervals

  auto partitionIntervals = iir::Interval::computePartition(intervals_v);
  if(ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
    std::reverse(partitionIntervals.begin(), partitionIntervals.end());
  return partitionIntervals;
}

void CudaCodeGen::generateCudaKernelCode(
    std::stringstream& ssSW, const std::shared_ptr<iir::StencilInstantiation> stencilInstantiation,
    const std::unique_ptr<iir::MultiStage>& ms, const CacheProperties& cacheProperties) {

  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(const auto& stage : iterateIIROver<iir::Stage>(*ms)) {
    maxExtents.merge(stage->getExtents());
  }

  const bool solveKLoopInParallel_ = solveKLoopInParallel(ms);

  // fields used in the stencil
  const auto& fields = ms->getFields();

  auto nonTempFields =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                            std::pair<int, iir::Field> const& p) {
                  return !stencilInstantiation->isTemporaryField(p.second.getAccessID());
                }));
  auto tempFieldsNonCached =
      makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                            std::pair<int, iir::Field> const& p) {
                  return stencilInstantiation->isTemporaryField(p.second.getAccessID()) &&
                         !cacheProperties.accessIsCached(p.second.getAccessID());
                }));

  const bool containsTemporary = !tempFieldsNonCached.empty();

  std::string fnDecl = "";
  if(containsTemporary)
    fnDecl = "template<typename TmpStorage>";
  fnDecl = fnDecl + "__global__ void";
  MemberFunction cudaKernel(fnDecl, buildCudaKernelName(stencilInstantiation, ms), ssSW);

  const auto& globalsMap = stencilInstantiation->getMetaData().globalVariableMap_;
  if(!globalsMap.empty()) {
    cudaKernel.addArg("globals globals_");
  }
  cudaKernel.addArg("const int isize");
  cudaKernel.addArg("const int jsize");
  cudaKernel.addArg("const int ksize");

  std::vector<std::string> strides = generateStrideArguments(
      nonTempFields, tempFieldsNonCached, *ms, *stencilInstantiation, FunctionArgType::callee);

  for(const auto strideArg : strides) {
    cudaKernel.addArg(strideArg);
  }

  // first we construct non temporary field arguments
  for(auto field : nonTempFields) {
    cudaKernel.addArg("gridtools::clang::float_type * const " +
                      stencilInstantiation->getNameFromAccessID((*field).second.getAccessID()));
  }

  // then the temporary field arguments
  for(auto field : tempFieldsNonCached) {
    cudaKernel.addArg(c_gt() + "data_view<TmpStorage>" +
                      stencilInstantiation->getNameFromAccessID((*field).second.getAccessID()) +
                      "_dv");
  }

  DAWN_ASSERT(fields.size() > 0);
  auto firstField = *(fields.begin());

  cudaKernel.startBody();
  cudaKernel.addComment("Start kernel");

  for(auto field : tempFieldsNonCached) {
    std::string fieldName =
        stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());

    cudaKernel.addStatement("gridtools::clang::float_type* " + fieldName + " = &" + fieldName +
                            "_dv(tmpBeginIIndex,tmpBeginJIndex,blockIdx.x,blockIdx.y,0)");
  }

  const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();

  generateIJCacheDecl(cudaKernel, stencilInstantiation, *ms, cacheProperties, blockSize);
  generateKCacheDecl(cudaKernel, *ms, cacheProperties);

  unsigned int ntx = blockSize[0];
  unsigned int nty = blockSize[1];
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
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize[0]),
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
          DAWN_ASSERT_MSG((jboundary_limit * paddedBoundary_ <= blockSize[0]),
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
    for(const auto& fieldInfo : ms->getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }
    DAWN_ASSERT(std::accumulate(dims.begin(), dims.end(), 0) != -3);
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

  if(cacheProperties.hasIJCaches()) {
    generateIJCacheIndexInit(cudaKernel, cacheProperties, blockSize);
  }

  if(containsTemporary) {
    generateTmpIndexInit(cudaKernel, ms, stencilInstantiation, cacheProperties);
  }

  // compute the partition of the intervals
  auto partitionIntervals = computePartitionOfIntervals(ms);

  DAWN_ASSERT(!partitionIntervals.empty());

  ASTStencilBody stencilBodyCXXVisitor(stencilInstantiation, fieldIndexMap, *ms, cacheProperties,
                                       blockSize);

  iir::Interval::IntervalLevel lastKCell{0, 0};
  lastKCell = advance(lastKCell, ms->getLoopOrder(), -1);

  bool firstInterval = true;
  for(auto interval : partitionIntervals) {

    // If execution is parallel we want to place the interval in a forward order
    if((solveKLoopInParallel_) && (interval.lowerBound() > interval.upperBound())) {
      interval.invert();
    }
    iir::IntervalDiff kmin{iir::IntervalDiff::RangeType::literal, 0};
    // if there is a jump between the last level of previous interval and the first level of this
    // interval, we advance the iterators

    iir::Interval::IntervalLevel nextLevel =
        computeNextLevelToProcess(interval, ms->getLoopOrder());
    auto jump = distance(lastKCell, nextLevel);
    if((std::abs(jump.value) != 1) || (jump.rangeType_ != iir::IntervalDiff::RangeType::literal)) {
      auto lastKCellp1 = advance(lastKCell, ms->getLoopOrder(), 1);
      kmin = distance(lastKCellp1, nextLevel);

      for(auto index : indexIterators) {
        if(index.second[2] && !kmin.null() && !((solveKLoopInParallel_) && firstInterval)) {
          cudaKernel.addComment("jump iterators to match the beginning of next interval");
          cudaKernel.addStatement("idx" + index.first + " += " +
                                  CodeGeneratorHelper::generateStrideName(2, index.second) + "*(" +
                                  intervalDiffToString(kmin, "ksize - 1") + ")");
        }
      }
      if(useTmpIndex(ms, stencilInstantiation, cacheProperties) && !kmin.null()) {
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
            step = "max(" + step + "," + " blockIdx.z * " + std::to_string(blockSize[2]) + ") * " +
                   CodeGeneratorHelper::generateStrideName(2, index.second);

            cudaKernel.addComment("jump iterators to match the intersection of beginning of next "
                                  "interval and the parallel execution block ");
            cudaKernel.addStatement("idx" + index.first + " += " + step);
          }
        }
      }
      if(useTmpIndex(ms, stencilInstantiation, cacheProperties)) {
        cudaKernel.addComment("jump tmp iterators to match the intersection of beginning of next "
                              "interval and the parallel execution block ");
        cudaKernel.addStatement("idx_tmp += max(" + intervalDiffToString(kmin, "ksize - 1") +
                                ", kstride_tmp * blockIdx.z * " + std::to_string(blockSize[2]) +
                                ")");
      }
    }
    // for each interval, we generate naive nested loops
    cudaKernel.addBlockStatement(
        makeKLoop("dom", blockSize, ms->getLoopOrder(), interval, solveKLoopInParallel_), [&]() {
          for(const auto& stagePtr : ms->getChildren()) {
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
                "if(iblock >= " + std::to_string(extent[0].Minus) +
                    " && iblock <= block_size_i -1 + " + std::to_string(extent[0].Plus) +
                    " && jblock >= " + std::to_string(extent[1].Minus) +
                    " && jblock <= block_size_j -1 + " + std::to_string(extent[1].Plus) + ")",
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
            if(intervalRequiresSync(interval, stage, ms)) {
              cudaKernel.addStatement("__syncthreads()");
            }
          }

          generateKCacheSlide(cudaKernel, cacheProperties, ms, interval);
          cudaKernel.addComment("increment iterators");
          std::string incStr =
              (ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward) ? "-=" : "+=";

          for(auto index : indexIterators) {
            if(index.second[2]) {
              cudaKernel.addStatement("idx" + index.first + incStr +
                                      CodeGeneratorHelper::generateStrideName(2, index.second));
            }
          }
          if(useTmpIndex(ms, stencilInstantiation, cacheProperties)) {
            cudaKernel.addStatement("idx_tmp " + incStr + " kstride_tmp");
          }
        });
    lastKCell = (ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward)
                    ? interval.lowerIntervalLevel()
                    : interval.upperIntervalLevel();
    firstInterval = false;
  }

  cudaKernel.commit();
}

void CudaCodeGen::generateKCacheSlide(MemberFunction& cudaKernel,
                                      const CacheProperties& cacheProperties,
                                      const std::unique_ptr<iir::MultiStage>& ms,
                                      const iir::Interval& interval) const {
  for(const auto& cachePair : ms->getCaches()) {
    const auto& cache = cachePair.second;
    if(!cacheProperties.isKCached(cache))
      continue;
    auto cacheInterval = cache.getInterval();
    DAWN_ASSERT(cacheInterval.is_initialized());
    if(!(*cacheInterval).overlaps(interval))
      continue;

    const int accessID = cache.getCachedFieldAccessID();
    auto vertExtent = cacheProperties.getKCacheVertExtent(accessID);
    auto cacheName = cacheProperties.getCacheName(accessID);

    for(int i = 0; i < -vertExtent.Minus + vertExtent.Plus; ++i) {
      if(ms->getLoopOrder() == iir::LoopOrderKind::LK_Backward) {
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
bool CudaCodeGen::intervalRequiresSync(const iir::Interval& interval, const iir::Stage& stage,
                                       const std::unique_ptr<iir::MultiStage>& ms) const {
  // if the stage is the last stage, it will require a sync (to ensure we sync before the write of a
  // previous stage at the next k level), but only if the stencil is not pure vertical and ij caches
  // are used after the last sync
  int lastStageID = -1;
  // we identified the last stage that required a sync
  int lastStageIDWithSync = -1;
  for(const auto& st : ms->getChildren()) {
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
  for(const auto& st : ms->getChildren()) {
    // we only activate the search to determine if IJ caches are used after last stage that was sync
    if(st->getStageID() == lastStageIDWithSync) {
      activateSearch = true;
    }
    if(!activateSearch)
      continue;
    const auto& fields = st->getFields();

    // If any IJ cache is used after the last synchronized stage,
    // we will need to sync again after the last stage of the vertical loop
    for(const auto& cache : ms->getCaches()) {
      if(cache.second.getCacheType() != iir::Cache::CacheTypeKind::IJ)
        continue;

      if(fields.count(cache.first)) {
        return true;
      }
    }
  }
  return false;
}

bool CudaCodeGen::solveKLoopInParallel(const std::unique_ptr<iir::MultiStage>& ms) const {
  iir::MultiInterval mInterval{computePartitionOfIntervals(ms)};
  return mInterval.contiguous() && (ms->getLoopOrder() == iir::LoopOrderKind::LK_Parallel);
}

iir::Interval::IntervalLevel
CudaCodeGen::computeNextLevelToProcess(const iir::Interval& interval,
                                       iir::LoopOrderKind loopOrder) const {
  iir::Interval::IntervalLevel intervalLevel;
  if(loopOrder == iir::LoopOrderKind::LK_Backward) {
    intervalLevel = interval.upperIntervalLevel();
  } else {
    intervalLevel = interval.lowerIntervalLevel();
  }
  return intervalLevel;
}
void CudaCodeGen::generateIJCacheIndexInit(MemberFunction& kernel,
                                           const CacheProperties& cacheProperties,
                                           const Array3ui blockSize) const {
  if(cacheProperties.isThereACommonCache()) {
    kernel.addStatement("int " +
                        cacheProperties.getCommonCacheIndexName(iir::Cache::CacheTypeKind::IJ) +
                        "= iblock + " + std::to_string(cacheProperties.getOffsetCommonCache(1)) +
                        " + (jblock + " + std::to_string(cacheProperties.getOffsetCommonCache(1)) +
                        ")*" + std::to_string(cacheProperties.getStrideCommonCache(1, blockSize)));
  }
}

bool CudaCodeGen::useTmpIndex(
    const std::unique_ptr<iir::MultiStage>& ms,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CacheProperties& cacheProperties) const {
  const auto& fields = ms->getFields();
  const bool containsTemporary =
      (find_if(fields.begin(), fields.end(), [&](const std::pair<int, iir::Field>& field) {
         const int accessID = field.second.getAccessID();
         // we dont need to initialize tmp indices for fields that are cached
         return stencilInstantiation->isTemporaryField(accessID) &&
                !cacheProperties.accessIsCached(accessID);
       }) != fields.end());

  return containsTemporary;
}

void CudaCodeGen::generateTmpIndexInit(
    MemberFunction& kernel, const std::unique_ptr<iir::MultiStage>& ms,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CacheProperties& cacheProperties) const {

  if(!useTmpIndex(ms, stencilInstantiation, cacheProperties))
    return;

  auto maxExtentTmps = computeTempMaxWriteExtent(*(ms->getParent()));
  kernel.addStatement("int idx_tmp = (iblock+" + std::to_string(-maxExtentTmps[0].Minus) +
                      ")*1 + (jblock+" + std::to_string(-maxExtentTmps[1].Minus) + ")*jstride_tmp");
}

int CudaCodeGen::paddedBoundary(int value) {
  return std::abs(value) <= 1 ? 1 : std::abs(value) <= 2 ? 2 : std::abs(value) <= 4 ? 4 : 8;
}
void CudaCodeGen::generateAllCudaKernels(
    std::stringstream& ssSW,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    DAWN_ASSERT(cachePropertyMap_.count(ms->getID()));
    generateCudaKernelCode(ssSW, stencilInstantiation, ms, cachePropertyMap_.at(ms->getID()));
  }
}

std::string CudaCodeGen::generateStencilInstantiation(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {
  using namespace codegen;

  std::stringstream ssSW;

  Namespace cudaNamespace("cuda", ssSW);

  // map from MS ID to cacheProperty
  for(const auto& ms : iterateIIROver<iir::MultiStage>(*(stencilInstantiation->getIIR()))) {
    cachePropertyMap_.emplace(ms->getID(), makeCacheProperties(ms, stencilInstantiation, 2));
  }

  generateAllCudaKernels(ssSW, stencilInstantiation);

  Class StencilWrapperClass(stencilInstantiation->getName(), ssSW);
  StencilWrapperClass.changeAccessibility("public");

  CodeGenProperties codeGenProperties = computeCodeGenProperties(stencilInstantiation.get());

  // generate code for base class of all the inner stencils
  Structure sbase = StencilWrapperClass.addStruct("sbase", "", "timer_cuda");
  auto baseCtr = sbase.addConstructor();
  baseCtr.addArg("std::string name");
  baseCtr.addInit("timer_cuda(name)");
  baseCtr.commit();
  MemberFunction gettime = sbase.addMemberFunction("double", "get_time");
  gettime.addStatement("return total_time()");
  gettime.commit();
  MemberFunction sbase_run = sbase.addMemberFunction("virtual void", "run");
  sbase_run.startBody();
  sbase_run.commit();
  MemberFunction sbase_sync = sbase.addMemberFunction("virtual void", "sync_storages");
  sbase_sync.startBody();
  sbase_sync.commit();

  MemberFunction sbaseVdtor = sbase.addMemberFunction("virtual", "~sbase");
  sbaseVdtor.startBody();
  sbaseVdtor.commit();
  sbase.commit();

  const auto& globalsMap = stencilInstantiation->getMetaData().globalVariableMap_;

  generateBoundaryConditionFunctions(StencilWrapperClass, stencilInstantiation);

  generateStencilClasses(stencilInstantiation, StencilWrapperClass, codeGenProperties);

  generateStencilWrapperMembers(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperCtr(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  if(!globalsMap.empty()) {
    generateGlobalsAPI(*stencilInstantiation, StencilWrapperClass, globalsMap, codeGenProperties);
  }

  generateStencilWrapperRun(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperSyncMethod(StencilWrapperClass, stencilInstantiation, codeGenProperties);

  generateStencilWrapperPublicMemberFunctions(StencilWrapperClass, codeGenProperties);

  StencilWrapperClass.commit();

  cudaNamespace.commit();

  // Remove trailing ';' as this is retained by Clang's Rewriter
  std::string str = ssSW.str();
  str[str.size() - 2] = ' ';

  return str;
}

void CudaCodeGen::generateStencilWrapperPublicMemberFunctions(
    Class& stencilWrapperClass, const CodeGenProperties& codeGenProperties) const {

  // Generate name getter
  stencilWrapperClass.addMemberFunction("std::string", "get_name")
      .isConst(true)
      .addStatement("return std::string(s_name)");

  std::vector<std::string> stencilMembers;

  for(const auto& stencilProp :
      codeGenProperties.getAllStencilProperties(StencilContext::SC_Stencil)) {
    stencilMembers.push_back("m_" + stencilProp.first);
  }

  // Generate stencil getter
  MemberFunction stencilGetter =
      stencilWrapperClass.addMemberFunction("std::vector<sbase*>", "getStencils");
  stencilGetter.addStatement("return " +
                             RangeToString(", ", "std::vector<sbase*>({", "})")(
                                 stencilMembers, [](const std::string& member) { return member; }));
  stencilGetter.commit();

  MemberFunction clearMeters = stencilWrapperClass.addMemberFunction("void", "reset_meters");
  clearMeters.startBody();
  std::string s = RangeToString("\n", "", "")(
      stencilMembers, [](const std::string& member) { return member + "->reset();"; });
  clearMeters << s;
  clearMeters.commit();
}

void CudaCodeGen::generateStencilClasses(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    Class& stencilWrapperClass, CodeGenProperties& codeGenProperties) const {
  // Generate stencils
  const auto& stencils = stencilInstantiation->getStencils();

  const auto& globalsMap = stencilInstantiation->getMetaData().globalVariableMap_;

  // generate the code for each of the stencils
  for(const auto& stencilPtr : stencils) {
    const auto& stencil = *stencilPtr;

    std::string stencilName = "stencil_" + std::to_string(stencil.getStencilID());
    auto stencilProperties =
        codeGenProperties.getStencilProperties(StencilContext::SC_Stencil, stencilName);

    if(stencil.isEmpty())
      continue;

    // fields used in the stencil
    const auto& StencilFields = stencil.getFields();

    auto nonTempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>([](
            std::pair<int, iir::Stencil::FieldInfo> const& p) { return !p.second.IsTemporary; }));
    auto tempFields = makeRange(
        StencilFields,
        std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
            [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));

    Structure stencilClass = stencilWrapperClass.addStruct(stencilName, "", "sbase");
    auto& paramNameToType = stencilProperties->paramNameToType_;

    for(auto fieldIt : nonTempFields) {
      paramNameToType.emplace(
          (*fieldIt).second.Name,
          getStorageType(stencilInstantiation->getFieldDimensionsMask((*fieldIt).first)));
    }

    for(auto fieldIt : tempFields) {
      paramNameToType.emplace((*fieldIt).second.Name, c_gtc().str() + "storage_t");
    }

    generateStencilClassMembers(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                                stencilProperties);

    stencilClass.changeAccessibility("public");

    generateStencilClassCtr(stencilClass, stencil, globalsMap, nonTempFields, tempFields,
                            stencilProperties);

    // virtual dtor
    MemberFunction stencilClassDtr = stencilClass.addDestructor();
    stencilClassDtr.startBody();
    stencilClassDtr.commit();

    // synchronize storages method
    MemberFunction syncStoragesMethod = stencilClass.addMemberFunction("void", "sync_storages", "");
    syncStoragesMethod.startBody();

    for(auto fieldIt : nonTempFields) {
      syncStoragesMethod.addStatement("m_" + (*fieldIt).second.Name + ".sync()");
    }

    syncStoragesMethod.commit();

    //
    // Run-Method
    //
    generateStencilRunMethod(stencilClass, stencil, stencilInstantiation, paramNameToType,
                             globalsMap);

    // Generate stencil getter
    stencilClass.addMemberFunction("sbase*", "get_stencil").addStatement("return this");
  }
}

void CudaCodeGen::generateStencilClassMembers(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto& paramNameToType = stencilProperties->paramNameToType_;

  stencilClass.addComment("Members");
  stencilClass.addComment("Temporary storages");
  addTempStorageTypedef(stencilClass, stencil);

  if(!globalsMap.empty()) {
    stencilClass.addMember("globals&", "m_globals");
  }

  stencilClass.addMember("const " + c_gtc() + "domain&", "m_dom");

  for(auto fieldIt : nonTempFields) {
    stencilClass.addMember(paramNameToType.at((*fieldIt).second.Name) + "&",
                           "m_" + (*fieldIt).second.Name);
  }

  addTmpStorageDeclaration(stencilClass, tempFields);
}
void CudaCodeGen::generateStencilClassCtr(
    Structure& stencilClass, const iir::Stencil& stencil, const sir::GlobalVariableMap& globalsMap,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& nonTempFields,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields,
    std::shared_ptr<StencilProperties> stencilProperties) const {

  auto stencilClassCtr = stencilClass.addConstructor();

  auto& paramNameToType = stencilProperties->paramNameToType_;

  stencilClassCtr.addArg("const " + c_gtc() + "domain& dom_");
  if(!globalsMap.empty()) {
    stencilClassCtr.addArg("globals& globals_");
  }

  for(auto fieldIt : nonTempFields) {
    std::string fieldName = (*fieldIt).second.Name;
    stencilClassCtr.addArg(paramNameToType.at(fieldName) + "& " + fieldName + "_");
  }

  stencilClassCtr.addInit("sbase(\"" + stencilClass.getName() + "\")");
  stencilClassCtr.addInit("m_dom(dom_)");

  if(!globalsMap.empty()) {
    stencilClassCtr.addInit("m_globals(globals_)");
  }

  for(auto fieldIt : nonTempFields) {
    stencilClassCtr.addInit("m_" + (*fieldIt).second.Name + "(" + (*fieldIt).second.Name + "_)");
  }

  addTmpStorageInit(stencilClassCtr, stencil, tempFields);
  stencilClassCtr.commit();
}

void CudaCodeGen::generateStencilWrapperCtr(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {

  const auto& globalsMap = stencilInstantiation->getMetaData().globalVariableMap_;

  // Generate stencil wrapper constructor
  auto StencilWrapperConstructor = stencilWrapperClass.addConstructor();
  StencilWrapperConstructor.addArg("const " + c_gtc() + "domain& dom");

  for(int fieldId : stencilInstantiation->getAPIFieldIDs()) {
    StencilWrapperConstructor.addArg(
        getStorageType(stencilInstantiation->getFieldDimensionsMask(fieldId)) + "& " +
        stencilInstantiation->getNameFromAccessID(fieldId));
  }

  const auto& stencils = stencilInstantiation->getStencils();

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const auto& StencilFields = stencil.getFields();

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    std::string initCtr = "m_" + stencilName + "(new " + stencilName;

    initCtr += "(dom";
    if(!globalsMap.empty()) {
      initCtr += ",m_globals";
    }

    for(const auto& fieldInfoPair : StencilFields) {
      const auto& fieldInfo = fieldInfoPair.second;
      if(fieldInfo.IsTemporary)
        continue;
      initCtr += "," + (stencilInstantiation->isAllocatedField(fieldInfo.field.getAccessID())
                            ? ("m_" + fieldInfo.Name)
                            : (fieldInfo.Name));
    }
    initCtr += ") )";
    StencilWrapperConstructor.addInit(initCtr);
  }

  if(stencilInstantiation->hasAllocatedFields()) {
    std::vector<std::string> tempFields;
    for(auto accessID : stencilInstantiation->getAllocatedFieldAccessIDs()) {
      tempFields.push_back(stencilInstantiation->getNameFromAccessID(accessID));
    }
    addTmpStorageInitStencilWrapperCtr(StencilWrapperConstructor, stencils, tempFields);
  }

  addBCFieldInitStencilWrapperCtr(StencilWrapperConstructor, codeGenProperties);

  StencilWrapperConstructor.commit();
}

void CudaCodeGen::generateStencilWrapperMembers(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    CodeGenProperties& codeGenProperties) const {

  const auto& globalsMap = stencilInstantiation->getMetaData().globalVariableMap_;

  stencilWrapperClass.addMember("static constexpr const char* s_name =",
                                Twine("\"") + stencilWrapperClass.getName() + Twine("\""));

  for(auto stencilPropertiesPair :
      codeGenProperties.stencilProperties(StencilContext::SC_Stencil)) {
    stencilWrapperClass.addMember("sbase*", "m_" + stencilPropertiesPair.second->name_);
  }

  stencilWrapperClass.changeAccessibility("public");
  stencilWrapperClass.addCopyConstructor(Class::Deleted);

  stencilWrapperClass.addComment("Members");

  //
  // Members
  //
  generateBCFieldMembers(stencilWrapperClass, stencilInstantiation, codeGenProperties);

  stencilWrapperClass.addComment("Stencil-Data");

  // Define allocated memebers if necessary
  if(stencilInstantiation->hasAllocatedFields()) {
    stencilWrapperClass.addMember(c_gtc() + "meta_data_t", "m_meta_data");

    for(int AccessID : stencilInstantiation->getAllocatedFieldAccessIDs())
      stencilWrapperClass.addMember(c_gtc() + "storage_t",
                                    "m_" + stencilInstantiation->getNameFromAccessID(AccessID));
  }

  if(!globalsMap.empty()) {
    stencilWrapperClass.addMember("globals", "m_globals");
  }
}

void CudaCodeGen::generateStencilWrapperRun(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  // Generate the run method by generate code for the stencil description AST
  MemberFunction RunMethod = stencilWrapperClass.addMemberFunction("void", "run", "");

  RunMethod.finishArgs();

  RunMethod.addStatement("sync_storages()");
  // generate the control flow code executing each inner stencil
  ASTStencilDesc stencilDescCGVisitor(stencilInstantiation, codeGenProperties);
  stencilDescCGVisitor.setIndent(RunMethod.getIndent());
  for(const auto& statement : stencilInstantiation->getStencilDescStatements()) {
    statement->ASTStmt->accept(stencilDescCGVisitor);
    RunMethod.addStatement(stencilDescCGVisitor.getCodeAndResetStream());
  }

  RunMethod.addStatement("sync_storages()");
  RunMethod.commit();
}

void CudaCodeGen::generateStencilWrapperSyncMethod(
    Class& stencilWrapperClass,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const CodeGenProperties& codeGenProperties) const {
  // Generate the run method by generate code for the stencil description AST
  MemberFunction syncMethod = stencilWrapperClass.addMemberFunction("void", "sync_storages");

  syncMethod.finishArgs();

  const auto& stencils = stencilInstantiation->getStencils();

  // add the ctr initialization of each stencil
  for(const auto& stencilPtr : stencils) {
    iir::Stencil& stencil = *stencilPtr;
    if(stencil.isEmpty())
      continue;

    const std::string stencilName =
        codeGenProperties.getStencilName(StencilContext::SC_Stencil, stencil.getStencilID());

    syncMethod.addStatement("m_" + stencilName + "->sync_storages()");
  }

  syncMethod.commit();
}

std::string CudaCodeGen::intervalDiffToString(iir::IntervalDiff intervalDiff,
                                              std::string maxRange) const {
  if(intervalDiff.rangeType_ == iir::IntervalDiff::RangeType::fullRange) {
    return maxRange + "+" + std::to_string(intervalDiff.value);
  }
  return std::to_string(intervalDiff.value);
}

void CudaCodeGen::generateStencilRunMethod(
    Structure& stencilClass, const iir::Stencil& stencil,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
    const std::unordered_map<std::string, std::string>& paramNameToType,
    const sir::GlobalVariableMap& globalsMap) const {
  MemberFunction StencilRunMethod = stencilClass.addMemberFunction("virtual void", "run", "");

  StencilRunMethod.startBody();

  StencilRunMethod.addComment("starting timers");
  StencilRunMethod.addStatement("start()");

  for(const auto& multiStagePtr : stencil.getChildren()) {
    StencilRunMethod.addStatement("{");

    const iir::MultiStage& multiStage = *multiStagePtr;
    const auto& cacheProperties = cachePropertyMap_.at(multiStagePtr->getID());

    bool solveKLoopInParallel_ = solveKLoopInParallel(multiStagePtr);

    const auto& fields = multiStage.getFields();

    auto nonTempFields =
        makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                              std::pair<int, iir::Field> const& p) {
                    return !stencilInstantiation->isTemporaryField(p.second.getAccessID());
                  }));

    auto tempFieldsNonCached =
        makeRange(fields, std::function<bool(std::pair<int, iir::Field> const&)>([&](
                              std::pair<int, iir::Field> const& p) {
                    return stencilInstantiation->isTemporaryField(p.second.getAccessID()) &&
                           !cacheProperties.accessIsCached(p.second.getAccessID());
                  }));

    // create all the data views
    for(auto fieldIt : nonTempFields) {
      // TODO have the same FieldInfo in ms level so that we dont need to query stencilInstantiation
      // all the time for name and IsTmpField
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*fieldIt).second.getAccessID());
      StencilRunMethod.addStatement(c_gt() + "data_view<" + paramNameToType.at(fieldName) + "> " +
                                    fieldName + "= " + c_gt() + "make_device_view(m_" + fieldName +
                                    ")");
    }
    for(auto fieldIt : tempFieldsNonCached) {
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*fieldIt).second.getAccessID());

      StencilRunMethod.addStatement(c_gt() + "data_view<tmp_storage_t> " + fieldName + "= " +
                                    c_gt() + "make_device_view(m_" + fieldName + ")");
    }

    DAWN_ASSERT(nonTempFields.size() > 0);

    iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
    for(const auto& stage : iterateIIROver<iir::Stage>(*multiStagePtr)) {
      maxExtents.merge(stage->getExtents());
    }

    StencilRunMethod.addStatement(
        "const unsigned int nx = m_dom.isize() - m_dom.iminus() - m_dom.iplus()");
    StencilRunMethod.addStatement(
        "const unsigned int ny = m_dom.jsize() - m_dom.jminus() - m_dom.jplus()");
    StencilRunMethod.addStatement(
        "const unsigned int nz = m_dom.ksize() - m_dom.kminus() - m_dom.kplus()");

    const auto blockSize = stencilInstantiation->getIIR()->getBlockSize();

    unsigned int ntx = blockSize[0];
    unsigned int nty = blockSize[1];

    StencilRunMethod.addStatement(
        "dim3 threads(" + std::to_string(ntx) + "," + std::to_string(nty) + "+" +
        std::to_string(maxExtents[1].Plus - maxExtents[1].Minus +
                       (maxExtents[0].Minus < 0 ? 1 : 0) + (maxExtents[0].Plus > 0 ? 1 : 0)) +
        ",1)");

    // number of blocks required
    StencilRunMethod.addStatement("const unsigned int nbx = (nx + " + std::to_string(ntx) +
                                  " - 1) / " + std::to_string(ntx));
    StencilRunMethod.addStatement("const unsigned int nby = (ny + " + std::to_string(nty) +
                                  " - 1) / " + std::to_string(nty));
    if(solveKLoopInParallel_) {
      StencilRunMethod.addStatement("const unsigned int nbz = (m_dom.ksize()+" +
                                    std::to_string(blockSize[2]) + "-1) / " +
                                    std::to_string(blockSize[2]));
    } else {
      StencilRunMethod.addStatement("const unsigned int nbz = 1");
    }
    StencilRunMethod.addStatement("dim3 blocks(nbx, nby, nbz)");
    std::string kernelCall =
        buildCudaKernelName(stencilInstantiation, multiStagePtr) + "<<<blocks, threads>>>(";

    if(!globalsMap.empty()) {
      kernelCall = kernelCall + "m_globals,";
    }

    // TODO enable const auto& below and/or enable use RangeToString
    std::string args;
    int idx = 0;
    for(auto field : nonTempFields) {
      const auto fieldName =
          stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());

      args = args + (idx == 0 ? "" : ",") + "(" + fieldName + ".data()+" + "m_" + fieldName +
             ".get_storage_info_ptr()->index(" + fieldName + ".template begin<0>(), " + fieldName +
             ".template begin<1>(),0 ))";
      ++idx;
    }
    DAWN_ASSERT(nonTempFields.size() > 0);
    for(auto field : tempFieldsNonCached) {
      args = args + "," + stencilInstantiation->getNameFromAccessID((*field).second.getAccessID());
    }

    std::vector<std::string> strides =
        generateStrideArguments(nonTempFields, tempFieldsNonCached, multiStage,
                                *stencilInstantiation, FunctionArgType::caller);

    DAWN_ASSERT(!strides.empty());

    kernelCall = kernelCall + "nx,ny,nz," + RangeToString(",", "", "")(strides) + "," + args + ")";

    StencilRunMethod.addStatement(kernelCall);

    StencilRunMethod.addStatement("}");
  }

  StencilRunMethod.addComment("stopping timers");
  StencilRunMethod.addStatement("pause()");

  StencilRunMethod.commit();
}

std::vector<std::string> CudaCodeGen::generateStrideArguments(
    const IndexRange<const std::unordered_map<int, iir::Field>>& nonTempFields,
    const IndexRange<const std::unordered_map<int, iir::Field>>& tempFields,
    const iir::MultiStage& ms, const iir::StencilInstantiation& stencilInstantiation,
    FunctionArgType funArg) const {

  std::unordered_set<std::string> processedDims;
  std::vector<std::string> strides;
  for(auto field : nonTempFields) {
    const auto fieldName = stencilInstantiation.getNameFromAccessID((*field).second.getAccessID());
    Array3i dims{-1, -1, -1};
    // TODO this is a hack, we need to have dimensions also at ms level
    for(const auto& fieldInfo : ms.getParent()->getFields()) {
      if(fieldInfo.second.field.getAccessID() == (*field).second.getAccessID()) {
        dims = fieldInfo.second.Dimensions;
        break;
      }
    }

    if(processedDims.count(CodeGeneratorHelper::indexIteratorName(dims))) {
      continue;
    }
    processedDims.emplace(CodeGeneratorHelper::indexIteratorName(dims));

    int usedDim = 0;
    for(int i = 0; i < dims.size(); ++i) {
      if(!dims[i])
        continue;
      if(!(usedDim++))
        continue;
      if(funArg == FunctionArgType::caller) {
        strides.push_back("m_" + fieldName + ".strides()[" + std::to_string(i) + "]");
      } else {
        strides.push_back("const int stride_" + CodeGeneratorHelper::indexIteratorName(dims) + "_" +
                          std::to_string(i));
      }
    }
  }
  if(!tempFields.empty()) {
    auto firstTmpField = **(tempFields.begin());
    std::string fieldName =
        stencilInstantiation.getNameFromAccessID(firstTmpField.second.getAccessID());
    if(funArg == FunctionArgType::caller) {
      strides.push_back("m_" + fieldName + ".get_storage_info_ptr()->template begin<0>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template begin<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<1>()," + "m_" +
                        fieldName + ".get_storage_info_ptr()->template stride<4>()");
    } else {
      strides.push_back("const int tmpBeginIIndex, const int tmpBeginJIndex, const int "
                        "jstride_tmp, const int kstride_tmp");
    }
  }

  return strides;
}

iir::Extents CudaCodeGen::computeTempMaxWriteExtent(iir::Stencil const& stencil) const {
  auto tempFields = makeRange(
      stencil.getFields(),
      std::function<bool(std::pair<int, iir::Stencil::FieldInfo> const&)>(
          [](std::pair<int, iir::Stencil::FieldInfo> const& p) { return p.second.IsTemporary; }));
  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  for(auto field : tempFields) {
    DAWN_ASSERT((*field).second.field.getWriteExtentsRB().is_initialized());
    maxExtents.merge(*((*field).second.field.getWriteExtentsRB()));
  }
  return maxExtents;
}
void CudaCodeGen::addTempStorageTypedef(Structure& stencilClass,
                                        iir::Stencil const& stencil) const {

  auto maxExtents = computeTempMaxWriteExtent(stencil);
  stencilClass.addTypeDef("tmp_halo_t")
      .addType("gridtools::halo< " + std::to_string(-maxExtents[0].Minus) + "," +
               std::to_string(-maxExtents[1].Minus) + ", 0, 0, " +
               std::to_string(getVerticalTmpHaloSize(stencil)) + ">");

  stencilClass.addTypeDef(tmpMetadataTypename_)
      .addType("storage_traits_t::storage_info_t< 0, 5, tmp_halo_t >");

  stencilClass.addTypeDef(tmpStorageTypename_)
      .addType("storage_traits_t::data_store_t< float_type, " + tmpMetadataTypename_ + ">");
}

void CudaCodeGen::addTmpStorageInit(
    MemberFunction& ctr, iir::Stencil const& stencil,
    IndexRange<const std::unordered_map<int, iir::Stencil::FieldInfo>>& tempFields) const {
  auto maxExtents = computeTempMaxWriteExtent(stencil);

  const auto blockSize = stencil.getParent()->getBlockSize();

  if(!(tempFields.empty())) {
    ctr.addInit(tmpMetadataName_ + "(" + std::to_string(blockSize[0]) + "+" +
                std::to_string(-maxExtents[0].Minus + maxExtents[0].Plus) + ", " +
                std::to_string(blockSize[1]) + "+" +
                std::to_string(-maxExtents[1].Minus + maxExtents[1].Plus) + ", (dom_.isize()+ " +
                std::to_string(blockSize[0]) + " - 1) / " + std::to_string(blockSize[0]) +
                ", (dom_.jsize()+ " + std::to_string(blockSize[1]) + " - 1) / " +
                std::to_string(blockSize[1]) + ", dom_.ksize() + 2 * " +
                std::to_string(getVerticalTmpHaloSize(stencil)) + ")");
    for(auto fieldIt : tempFields) {
      ctr.addInit("m_" + (*fieldIt).second.Name + "(" + tmpMetadataName_ + ")");
    }
  }
}

std::unique_ptr<TranslationUnit> CudaCodeGen::generateCode() {
  DAWN_LOG(INFO) << "Starting code generation for GTClang ...";

  // Generate code for StencilInstantiations
  std::map<std::string, std::string> stencils;
  for(const auto& nameStencilCtxPair : context_->getStencilInstantiationMap()) {
    std::shared_ptr<iir::StencilInstantiation> origSI = nameStencilCtxPair.second;
    // TODO the clone seems to be broken
    //    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI->clone();
    std::shared_ptr<iir::StencilInstantiation> stencilInstantiation = origSI;

    PassInlining inliner(PassInlining::InlineStrategyKind::IK_Precomputation);

    inliner.run(stencilInstantiation);

    std::string code = generateStencilInstantiation(stencilInstantiation);
    if(code.empty())
      return nullptr;
    stencils.emplace(nameStencilCtxPair.first, std::move(code));
  }

  std::string globals = generateGlobals(context_->getSIR(), "cuda");

  std::vector<std::string> ppDefines;
  auto makeDefine = [](std::string define, int value) {
    return "#define " + define + " " + std::to_string(value);
  };

  ppDefines.push_back(makeDefine("GRIDTOOLS_CLANG_GENERATED", 1));
  ppDefines.push_back("#define GRIDTOOLS_CLANG_BACKEND_T CUDA");
  //==============------------------------------------------------------------------------------===
  // BENCHMARKTODO: since we're importing two cpp files into the benchmark API we need to set
  // these
  // variables also in the naive code-generation in order to not break it. Once the move to
  // different TU's is completed, this is no longer necessary.
  // [https://github.com/MeteoSwiss-APN/gtclang/issues/32]
  //==============------------------------------------------------------------------------------===
  CodeGen::addMplIfdefs(ppDefines, 30, context_->getOptions().MaxHaloPoints);

  generateBCHeaders(ppDefines);

  DAWN_LOG(INFO) << "Done generating code";

  // TODO missing the BC
  return make_unique<TranslationUnit>(context_->getSIR()->Filename, std::move(ppDefines),
                                      std::move(stencils), std::move(globals));
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
