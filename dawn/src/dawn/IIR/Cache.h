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

#include "dawn/IIR/Interval.h"
#include "dawn/Support/HashCombine.h"
#include <optional>
#include <string>

namespace dawn {
namespace iir {

/// @brief Cache specification of gridtools
/// @ingroup optimizer
class Cache {
public:
  struct window {
    int m_m, m_p;
    std::string toString() const {
      return std::string("[") + std::to_string(m_m) + "," + std::to_string(m_p) + "]";
    }
  };

  /// @brief Available cache types
  enum class CacheType {
    IJ,  ///< IJ caches require synchronization capabilities, as different (i,j) grid points are
         ///  processed by parallel cores. GPU backend keeps them in shared memory
    K,   ///< Processing of all the K elements is done by same thread, so resources for K caches can
         ///  be private and do not require synchronization. GPU backend uses registers.
    IJK, ///< IJK caches is an extension to 3rd dimension of IJ caches. GPU backend uses shared
         ///  memory
    bypass ///< bypass the cache for read only parameters
  };

  /// @brief IO policies of the cache
  enum class IOPolicy {
    unknown,        ///< Not yet set
    fill_and_flush, ///< Read values from the cached field and write the result back
    fill,           ///< Read values form the cached field but do not write back
    flush,          ///< Write values back the the cached field but do not read in
    epflush,        ///< End point cache flush: indicates a flush only at the end point
                    ///  of the interval being cached
    bpfill,         ///< Begin point cache fill: indicates a fill only at the begin point
                    ///  of the interval being cached
    local           ///< Local only cache, neither read nor write the the cached field
  };

  Cache(CacheType type, IOPolicy policy, int AccessID, std::optional<Interval> const& interval,
        std::optional<Interval> const& enclosingAccessedInterval, std::optional<window> const& w);

  /// @brief Get the AccessID of the field
  int getCachedFieldAccessID() const;

  json::json jsonDump() const;

  /// @brief Get the type of cache
  CacheType getType() const;
  std::string getTypeAsString() const;

  /// @brief Get the I/O policy of the cache
  IOPolicy getIOPolicy() const;
  std::string getIOPolicyAsString() const;

  /// @brief Get the interval of the iteration space from where the cache was accessed
  std::optional<Interval> getInterval() const;

  /// @brief returns a crop of the interval with the window of the cache (according to the specified
  /// bound)
  Interval getWindowInterval(Interval::Bound bound) const;

  /// @brief Get the enclosing of the iteration space interval and the accesses extent
  std::optional<Interval> getEnclosingAccessedInterval() const;

  /// @brief determines if the cache specification requires a window
  bool requiresWindow() const;

  bool requiresMemMemoryAccess() const;

  /// @name Comparison operator
  /// @{
  bool operator==(const Cache& other) const {
    return (AccessID_ == other.AccessID_ && type_ == other.type_ && policy_ == other.policy_);
  }
  bool operator!=(const Cache& other) const { return !(*this == other); }
  /// @}

  std::optional<window> const& getWindow() const { return window_; }

private:
  CacheType type_;
  IOPolicy policy_;
  int AccessID_;
  std::optional<Interval> interval_;
  std::optional<Interval> enclosingAccessedInterval_;
  std::optional<window> window_;
};

std::ostream& operator<<(std::ostream& os, Cache::window const& w);
bool operator==(const Cache::window& first, const Cache::window& second);

} // namespace iir
} // namespace dawn

namespace std {

template <>
struct hash<dawn::iir::Cache> {
  size_t operator()(const dawn::iir::Cache& cache) const {
    std::size_t seed = 0;
    dawn::hash_combine(seed, cache.getCachedFieldAccessID(), static_cast<int>(cache.getIOPolicy()),
                       static_cast<int>(cache.getType()));
    return seed;
  }
};

} // namespace std
