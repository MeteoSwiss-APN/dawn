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

#include "dawn/CodeGen/Cuda/CacheProperties.h"
#include "dawn/CodeGen/Cuda/CodeGeneratorHelper.h"
#include "dawn/IIR/IIRNodeIterator.h"
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {
namespace cuda {

CacheProperties
makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                    const int maxRedundantLines) {

  iir::Extents maxExtents(dawn::ast::cartesian_{}, 0, 0, 0, 0, 0, 0);
  std::set<int> accessIDs;
  std::unordered_map<int, iir::Extents> specialCaches;
  for(const auto& cacheP : ms->getCaches()) {
    const int accessID = cacheP.first;
    auto v_extents = ms->getField(accessID).getExtentsRB().verticalExtent();
    auto h_extents = dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(
        ms->getField(accessID).getExtentsRB().horizontalExtent());

    bool exceeds = false;

    const int numDimensions = 3;
    std::array<int, numDimensions> extentsMinus(
        {h_extents.iMinus(), h_extents.jMinus(), v_extents.minus()});
    std::array<int, numDimensions> extentsPlus(
        {h_extents.iPlus(), h_extents.jPlus(), v_extents.plus()});

    for(int i = 0; i < numDimensions; ++i) {
      if(extentsMinus[i] < -maxRedundantLines || extentsPlus[i] > maxRedundantLines) {
        exceeds = true;
      }
    }

    std::array<int, numDimensions> maxExtentsMinus({0, 0, 0});
    std::array<int, numDimensions> maxExtentsPlus({0, 0, 0});

    for(int i = 0; i < numDimensions; ++i) {
      maxExtentsMinus[i] =
          std::max(-maxRedundantLines, std::min(extentsMinus[i], maxExtentsMinus[i]));
      maxExtentsPlus[i] = std::min(maxRedundantLines, std::max(extentsPlus[i], maxExtentsPlus[i]));
    }

    if(!exceeds) {
      accessIDs.insert(accessID);
    } else {
      specialCaches.emplace(accessID, iir::Extents(ast::cartesian_{}, extentsMinus[0],
                                                   extentsPlus[0], extentsMinus[1], extentsPlus[1],
                                                   extentsMinus[2], extentsPlus[2]));
    }
  }
  return CacheProperties{ms, std::move(accessIDs), maxExtents, std::move(specialCaches),
                         stencilInstantiation};
}

CacheProperties::CacheProperties(
    const std::unique_ptr<iir::MultiStage>& ms, const std::set<int>& accessIDsCommonCache,
    iir::Extents extents, const std::unordered_map<int, iir::Extents>& specialCaches,
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation)
    : ms_(ms), accessIDsCommonCache_(accessIDsCommonCache), extents_(extents),
      specialCaches_(specialCaches), stencilInstantiation_(stencilInstantiation),
      metadata_(stencilInstantiation->getMetaData()) {}

std::string CacheProperties::getCacheName(int accessID) const {

  const auto& cache = ms_->getCache(accessID);
  if(cache.getCacheType() == iir::Cache::CacheTypeKind::IJ)
    return metadata_.getFieldNameFromAccessID(cache.getCachedFieldAccessID()) + "_ijcache";
  else if(cache.getCacheType() == iir::Cache::CacheTypeKind::K)
    return metadata_.getFieldNameFromAccessID(cache.getCachedFieldAccessID()) + "_kcache";

  dawn_unreachable("Unknown cache for code generation");
}

bool CacheProperties::isCached(const int accessID) const {
  return ms_->getCaches().count(accessID);
}

bool CacheProperties::isIJCached(const int accessID) const {
  return isCached(accessID) &&
         (ms_->getCache(accessID).getCacheType() == iir::Cache::CacheTypeKind::IJ);
}

int CacheProperties::getKCacheIndex(const int accessID, const int offset) const {
  return getKCacheCenterOffset(accessID) + offset;
}

bool CacheProperties::requiresFill(const iir::Cache& cache) {
  return ((cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill) ||
          (cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::bpfill) ||
          (cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill_and_flush));
}

int CacheProperties::getKCacheCenterOffset(const int accessID) const {
  auto ext = ms_->getKCacheVertExtent(accessID);
  return -ext.Minus;
}

bool CacheProperties::isKCached(const int accessID) const {
  return isCached(accessID) && isKCached(ms_->getCache(accessID));
}

bool CacheProperties::isKCached(const iir::Cache& cache) const {
  bool solveKLoopInParallel_ = CodeGeneratorHelper::solveKLoopInParallel(ms_);
  if(cache.getCacheType() != iir::Cache::CacheTypeKind::K) {
    return false;
  }

  return ((cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::local) || !solveKLoopInParallel_);
}

bool CacheProperties::hasIJCaches() const {
  for(const auto& cacheP : ms_->getCaches()) {
    const iir::Cache& cache = cacheP.second;
    if(cache.getCacheType() != iir::Cache::CacheTypeKind::IJ)
      continue;
    return true;
  }
  return false;
}

bool CacheProperties::accessIsCached(const int accessID) const {
  return ms_->isCached(accessID) && (isIJCached(accessID) || isKCached(accessID));
}

iir::Extents CacheProperties::getCacheExtent(int accessID) const {
  if(isCommonCache(accessID)) {
    return extents_;
  } else {
    return specialCaches_.at(accessID);
  }
}

int CacheProperties::getStride(int accessID, int dim, Array3ui blockSize) const {
  auto h_extents = dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(
      getCacheExtent(accessID).horizontalExtent());
  return getStrideImpl(dim, blockSize, h_extents.iMinus(), h_extents.iPlus());
}

int CacheProperties::getStrideCommonCache(int dim, Array3ui blockSize) const {
  auto h_extents =
      dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extents_.horizontalExtent());
  return getStrideImpl(dim, blockSize, h_extents.iMinus(), h_extents.iPlus());
} // namespace cuda

int CacheProperties::getStrideImpl(int dim, Array3ui blockSize, int Minus, int Plus) const {
  if(dim == 0) {
    return 1;
  } else if(dim == 1) {
    return blockSize[0] - Minus + Plus;
  } else {
    dawn_unreachable("error");
  }
}

int CacheProperties::getOffsetBeginIJCache(int accessID, int dim) const {
  DAWN_ASSERT(dim <= 2);
  auto h_extents = dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(
      getCacheExtent(accessID).horizontalExtent());
  if(dim == 0) {
    return -h_extents.iMinus();
  } else {
    return -h_extents.jMinus();
  }
}

int CacheProperties::getOffsetCommonIJCache(int dim) const {
  DAWN_ASSERT(dim <= 2);
  auto h_extents =
      dawn::iir::extent_cast<dawn::iir::CartesianExtent const&>(extents_.horizontalExtent());
  if(dim == 0) {
    return -h_extents.iMinus();
  } else {
    return -h_extents.jMinus();
  }
}

std::string CacheProperties::getCommonCacheIndexName(iir::Cache::CacheTypeKind cacheType) const {
  if(cacheType == iir::Cache::CacheTypeKind::IJ) {
    return "ijcacheindex";
  }
  dawn_unreachable("unknown cache type");
}

bool CacheProperties::isThereACommonCache() const { return !(accessIDsCommonCache_.empty()); }

} // namespace cuda
} // namespace codegen
} // namespace dawn
