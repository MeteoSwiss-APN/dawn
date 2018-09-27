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
#include <set>
#include "dawn/IIR/Extents.h"
#include "dawn/IIR/StencilInstantiation.h"

namespace dawn {
namespace codegen {
namespace cuda {

struct CacheProperties {
  const std::unique_ptr<iir::MultiStage>& ms_;
  std::set<int> accessIDsCommonCache_;
  iir::Extents extents_;
  std::unordered_map<int, iir::Extents> specialCaches_;

  bool isCommonCache(int accessID) const { return accessIDsCommonCache_.count(accessID); }

  iir::Extents getCacheExtent(int accessID) const;

  std::string
  getCacheName(int accessID,
               const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) const;

  int getStride(int accessID, int dim, Array3ui blockSize) const;
  int getOffset(int accessID, int dim) const;
};

CacheProperties makeCacheProperties(const std::unique_ptr<iir::MultiStage>& ms,
                                    const int maxRedundantLines);

} // namespace cuda
} // namespace codegen
} // namespace dawn

#endif
