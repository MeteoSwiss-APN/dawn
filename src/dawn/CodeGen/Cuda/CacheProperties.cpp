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
#include "dawn/IIR/MultiStage.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {
namespace cuda {

CacheProperties
makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                    const int maxRedundantLines) {

  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  std::set<int> accessIDs;
  std::unordered_map<int, iir::Extents> specialCaches;
  for(const auto& cacheP : ms->getCaches()) {
    const int accessID = cacheP.first;
    auto extents = ms->getField(accessID).getExtentsRB();
    bool exceeds = false;
    for(int i = 0; i < 3; ++i) {
      if(extents[i].Minus < -maxRedundantLines || extents[i].Plus > maxRedundantLines) {
        exceeds = true;
      }
      maxExtents[i].Minus =
          std::max(-maxRedundantLines, std::min(extents[i].Minus, maxExtents[i].Minus));
      maxExtents[i].Plus =
          std::min(maxRedundantLines, std::max(extents[i].Plus, maxExtents[i].Plus));
    }
    if(!exceeds) {
      accessIDs.insert(accessID);
    } else {
      specialCaches.emplace(accessID, extents);
    }
  }
  return CacheProperties{ms, std::move(accessIDs), maxExtents, std::move(specialCaches),
                         stencilInstantiation};
}

std::string CacheProperties::getCacheName(int accessID) const {

  const auto& cache = ms_->getCache(accessID);
  if(cache.getCacheType() == iir::Cache::CacheTypeKind::IJ)
    return stencilInstantiation_->getNameFromAccessID(cache.getCachedFieldAccessID()) + "_ijcache";
  else if(cache.getCacheType() == iir::Cache::CacheTypeKind::K)
    return stencilInstantiation_->getNameFromAccessID(cache.getCachedFieldAccessID()) + "_kcache";

  dawn_unreachable("Unknown cache for code generation");
}

bool CacheProperties::isCached(const int accessID) const {
  return ms_->getCaches().count(accessID);
}

bool CacheProperties::isIJCached(const int accessID) const {
  return isCached(accessID) &&
         (ms_->getCache(accessID).getCacheType() == iir::Cache::CacheTypeKind::IJ);
}

iir::Extent CacheProperties::getKCacheVertExtent(const int accessID) const {
  const auto& field = ms_->getField(accessID);
  auto vertExtent = field.getExtents()[2];
  return vertExtent;
}

int CacheProperties::getKCacheIndex(const int accessID, const int offset) const {
  return getKCacheCenterOffset(accessID) + offset;
}

bool CacheProperties::requiresFill(const iir::Cache& cache) {
  return ((cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill) ||
          (cache.getCacheIOPolicy() == iir::Cache::CacheIOPolicy::fill_and_flush));
}

int CacheProperties::getKCacheCenterOffset(const int accessID) const {
  auto ext = getKCacheVertExtent(accessID);
  return -ext.Minus;
}

bool CacheProperties::isKCached(const int accessID) const {
  return isCached(accessID) && isKCached(ms_->getCache(accessID));
}

bool CacheProperties::isKCached(const iir::Cache& cache) const {
  return (cache.getCacheType() == iir::Cache::CacheTypeKind::K);
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
  auto extents = getCacheExtent(accessID);
  return getStrideImpl(dim, blockSize, extents);
}

int CacheProperties::getStrideCommonCache(int dim, Array3ui blockSize) const {
  return getStrideImpl(dim, blockSize, extents_);
}

int CacheProperties::getStrideImpl(int dim, Array3ui blockSize, const iir::Extents& extents) const {
  if(dim == 0) {
    return 1;
  } else if(dim == 1) {
    return blockSize[0] - extents[0].Minus + extents[0].Plus;
  } else {
    dawn_unreachable("error");
  }
}

// TODO rename this
int CacheProperties::getOffset(int accessID, int dim) const {
  auto extents = getCacheExtent(accessID);
  return -extents[dim].Minus;
}

int CacheProperties::getOffsetCommonCache(int dim) const { return -extents_[dim].Minus; }

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
