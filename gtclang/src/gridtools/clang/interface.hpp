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

#pragma once

namespace gtclang {

template <typename T>
void e_fieldType(...);
template <typename T>
void v_fieldType(...);
template <typename T>
void c_fieldType(...);
void meshType(...);
template <typename Tag, typename T>
using edge_field_t = decltype(e_fieldType<T>(Tag{}));
template <typename Tag, typename T>
using vertex_field_t = decltype(v_fieldType<T>(Tag{}));
template <typename Tag, typename T>
using cell_field_t = decltype(c_fieldType<T>(Tag{}));
template <typename Tag>
using mesh_t = decltype(meshType(Tag{}));

} // namespace gtclang
