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

namespace gridtools {
namespace dawn {

template <class DataView>
struct param_wrapper {
  using offset_t = std::array<int, DataView::storage_info_t::ndims>;
  DataView dview_;
  offset_t offsets_;

  param_wrapper(DataView dview, std::array<int, DataView::storage_info_t::ndims> offsets)
      : dview_(dview), offsets_(offsets) {}

  void addOffset(offset_t offsets) { offsets_ = offsets_ + offsets; }

  param_wrapper<DataView> cloneWithOffset(offset_t offset) const {
    param_wrapper<DataView> res(*this);
    res.addOffset(offset);
    return res;
  }
};

} // namespace dawn
} // namespace gridtools
