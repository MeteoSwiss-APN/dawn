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
void cellFieldType(...);
template <typename T>
void edgeFieldType(...);
template <typename T>
void vertexFieldType(...);
void meshType(...);
template <typename Tag, typename T>
using cell_field_t = decltype(cellFieldType<T>(Tag{}));
template <typename Tag, typename T>
using edge_field_t = decltype(edgeFieldType<T>(Tag{}));
template <typename Tag, typename T>
using vertex_field_t = decltype(vertexFieldType<T>(Tag{}));
template <typename Tag>
using mesh_t = decltype(meshType(Tag{}));

// generic deref, specialize if needed
template <typename Tag, typename LocationType>
auto const& deref(Tag, LocationType const& l) {
  return l;
}
} // namespace gtclang
