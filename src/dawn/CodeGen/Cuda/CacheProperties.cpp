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

#include "dawn/CodeGen/Cuda/CacheProperties.hpp"

namespace dawn {
namespace codegen {
namespace cuda {

CacheProperties makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                                    const int maxRedundantLines) {

  iir::Extents maxExtents{0, 0, 0, 0, 0, 0};
  std::set<int> accessIDs;
  std::unordered_map<int, iir::Extents> specialCaches;
  for(const auto& cacheP : ms->getCaches()) {
    const int accessID = cacheP.first;
    auto extents = ms->getField(accessID).getExtentsRB();
    bool exceeds = false;
    for(int i = 0; i < 3; ++i) {
      if(extents[i].Minus < maxRedundantLines || extents[i].Plus > maxRedundantLines) {
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
  return CacheProperties{ms, std::move(accessIDs), maxExtents, std::move(specialCaches)};
}

std::string CacheProperties::getCacheName(
    int accessID, const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const {

  const auto& cache = ms_->getCache(accessID);
  if(cache.getCacheType() == iir::Cache::CacheTypeKind::IJ)
    return stencilInstantiation->getNameFromAccessID(cache.getCachedFieldAccessID()) + "_ijcache";
  else if(cache.getCacheType() == iir::Cache::CacheTypeKind::K)
    return stencilInstantiation->getNameFromAccessID(cache.getCachedFieldAccessID()) + "_kcache";

  dawn_unreachable("Unknown cache for code generation");
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
  if(dim == 0)
    return 1;
  else if(dim == 1)
    return blockSize[1] - extents[1].Minus + extents[1].Plus;
  else
    dawn_unreachable("error");
}

int CacheProperties::getOffset(int accessID, int dim) const {
  auto extents = getCacheExtent(accessID);
  return -extents[dim].Minus;
}

} // namespace cuda
} // namespace codegen
} // namespace dawn
