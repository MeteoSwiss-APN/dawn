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
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/MultiStage.h"
#include <memory>
#include <set>

namespace dawn {
namespace iir {
class StencilInstantiation;
class MultiStage;
} // namespace iir

namespace codegen {
namespace cuda {

struct CacheProperties {
  const std::unique_ptr<iir::MultiStage>& ms_;
  std::set<int> accessIDsCommonCache_;
  iir::Extents extents_;
  std::unordered_map<int, iir::Extents> specialCaches_;
  const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation_;
  const iir::StencilMetaInformation& metadata_;

  CacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                  const std::set<int>& accessIDsCommonCache_, iir::Extents extents_,
                  const std::unordered_map<int, iir::Extents>& specialCaches_,
                  const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation_);

  /// @brief in order to minimize the integer iterator operations, IJ caches are allocated with a
  /// common size (up to a predefined maximum) so that the can reuse the same iterator. This
  /// function return true if the cache falls into this size, so that we can use this common
  /// iterator
  bool isCommonCache(int accessID) const { return accessIDsCommonCache_.count(accessID); }

  /// @brief returns the extent of the cache
  iir::Extents const& getCacheExtent(int accessID) const;

  /// @brief returns the name of a cache (for code generation) given its accessID
  std::string getCacheName(int accessID) const;

  /// @brief returns the stride in a diven dimension of an IJ cache
  int getStride(int accessID, int dim, Array3ui blockSize) const;

  /// @brief returns the stride in a diven dimension of common cache @see isCommonCache
  int getStrideCommonCache(int dim, Array3ui blockSize) const;

  /// @brief returns the offset in certain dimension of the beginning of a cache
  int getOffsetBeginIJCache(int accessID, int dim) const;

  /// @brief returns the offset (in a kcache ring buffer) of the beginning of a cache for a common
  /// cache @see isCommonCache
  int getOffsetCommonIJCache(int dim) const;

  /// @brief returns the name of the index of a common cache @see isCommonCache
  std::string getCommonCacheIndexName(iir::Cache::CacheType cacheType) const;
  /// @brief true if there is at least one common cache (@see isCommonCache)
  bool isThereACommonCache() const;

  /// @brief returns true of the accessID is cached
  bool accessIsCached(const int accessID) const;

  /// @brief returns true if there are IJ caches registered
  bool hasIJCaches() const;
  /// @brief true if the accessID should be IJ cached
  bool isIJCached(const int accessID) const;
  /// @brief true if the accessID should be K cached
  bool isKCached(const int accessID) const;

  /// @brief true if the accessID should be cached
  bool isCached(const int accessID) const;

  /// @brief returns the offset (in the ring buffer) of the center of the kcache
  int getKCacheCenterOffset(const int accessID) const;
  /// @brief if the cache is kcached
  bool isKCached(const iir::Cache& cache) const;

  /// @brief returns the index (of the ring buffer) for a stencil access with a vertical offset
  int getKCacheIndex(const int accessID, const int offset) const;

  /// @brief true if the cache requires a fill
  static bool requiresFill(const iir::Cache& cache);
};

CacheProperties
makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation,
                    const int maxRedundantLines);

} // namespace cuda
} // namespace codegen
} // namespace dawn
