//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GRIDTOOLS_CLANG_PARAMWRAPPER_HPP
#define GRIDTOOLS_CLANG_PARAMWRAPPER_HPP

namespace gridtools {
    namespace clang {

        template < class DataView >
        struct ParamWrapper {
            using offset_t = std::array< int, DataView::storage_info_t::ndims >;
            DataView dview_;
            offset_t offsets_;

            ParamWrapper(DataView dview, std::array< int, DataView::storage_info_t::ndims > offsets)
                : dview_(dview), offsets_(offsets) {}

            void addOffset(offset_t offsets) { offsets_ = offsets_ + offsets; }

            ParamWrapper< DataView > cloneWithOffset(offset_t offset) const {
                ParamWrapper< DataView > res(*this);
                res.addOffset(offset);
                return res;
            }
        };

    } // namespace clang
} // namespace gridtools

#endif
