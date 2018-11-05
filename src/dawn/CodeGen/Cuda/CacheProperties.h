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

#ifndef DAWN_CODEGEN_CACHEPROPERTIES_H
#define DAWN_CODEGEN_CACHEPROPERTIES_H
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/MultiStage.h"
#include <memory>
#include <set>

namespace dawn {
namespace iir {
class StencilInstantiation;
class MultiStage;
}

namespace codegen {
namespace cuda {

struct CacheProperties {
  const std::unique_ptr<iir::MultiStage>& ms_;
  std::set<int> accessIDsCommonCache_;
  iir::Extents extents_;
  std::unordered_map<int, iir::Extents> specialCaches_;
  const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation_;

  bool isCommonCache(int accessID) const { return accessIDsCommonCache_.count(accessID); }

  iir::Extents getCacheExtent(int accessID) const;

  std::string getCacheName(int accessID) const;

  int getStride(int accessID, int dim, Array3ui blockSize) const;
  int getStrideCommonCache(int dim, Array3ui blockSize) const;
  int getOffset(int accessID, int dim) const;
  int getOffsetCommonCache(int dim) const;
  std::string getCommonCacheIndexName(iir::Cache::CacheTypeKind cacheType) const;
  bool isThereACommonCache() const;
  bool accessIsCached(const int accessID) const;
  bool hasIJCaches() const;
  bool isIJCached(const int accessID) const;
  bool isKCached(const int accessID) const;
  bool isCached(const int accessID) const;
  iir::Extent getKCacheVertExtent(const int accessID) const;
  int getKCacheCenterOffset(const int accessID) const;
  bool isKCached(const iir::Cache& cache) const;
  int getKCacheIndex(const int accessID, const int offset) const;
  static bool requiresFill(const iir::Cache& cache);
  static bool requiresFlush(const iir::Cache& cache);

private:
  int getStrideImpl(int dim, Array3ui blockSize, const iir::Extents& extents) const;
};

CacheProperties
makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                    const int maxRedundantLines);

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
