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

  iir::Extents maxExtents{ast::cartesian};

  std::set<int> accessIDs;
  std::unordered_map<int, iir::Extents> specialCaches;
  for(const auto& cacheP : ms->getCaches()) {
    const int accessID = cacheP.first;
    auto originalExtent = ms->getField(accessID).getExtentsRB();
    iir::Extents limitedExtent = originalExtent.limit(
        {ast::cartesian, -maxRedundantLines, maxRedundantLines, -maxRedundantLines,
         maxRedundantLines, -maxRedundantLines, maxRedundantLines});
    maxExtents.merge(limitedExtent);

    if(limitedExtent == originalExtent) {
      accessIDs.insert(accessID);
    } else {
      specialCaches.emplace(accessID, originalExtent);
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
  return -ext.minus();
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

iir::Extents const& CacheProperties::getCacheExtent(int accessID) const {
  if(isCommonCache(accessID)) {
    return extents_;
  } else {
    return specialCaches_.at(accessID);
  }
}

namespace {
int getStrideImpl(int dim, Array3ui blockSize, iir::Extents const& extents) {
  auto const& hExtents = iir::extent_cast<iir::CartesianExtent const&>(extents.horizontalExtent());
  if(dim == 0) {
    return 1;
  } else if(dim == 1) {
    return blockSize[0] - hExtents.iMinus() + hExtents.iPlus();
  } else {
    dawn_unreachable("error");
  }
}
} // namespace

int CacheProperties::getStride(int accessID, int dim, Array3ui blockSize) const {
  return getStrideImpl(dim, blockSize, getCacheExtent(accessID));
}

int CacheProperties::getStrideCommonCache(int dim, Array3ui blockSize) const {
  return getStrideImpl(dim, blockSize, extents_);
}

int CacheProperties::getOffsetBeginIJCache(int accessID, int dim) const {
  DAWN_ASSERT(dim <= 1);
  auto const& extent = getCacheExtent(accessID);
  auto const& hExtents = iir::extent_cast<iir::CartesianExtent const&>(extent.horizontalExtent());
  return dim == 0 ? -hExtents.iMinus() : -hExtents.jMinus();
}

int CacheProperties::getOffsetCommonIJCache(int dim) const {
  DAWN_ASSERT(dim <= 1);
  auto const& hExtents = iir::extent_cast<iir::CartesianExtent const&>(extents_.horizontalExtent());
  return dim == 0 ? -hExtents.iMinus() : -hExtents.jMinus();
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
