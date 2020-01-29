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

#include "defs.hpp"

namespace dawn {

template <typename T>
void cellFieldType(...);
template <typename T>
void edgeFieldType(...);
template <typename T>
void vertexFieldType(...);

template <typename T>
void sparseCellFieldType(...);
template <typename T>
void sparseEdgeFieldType(...);
template <typename T>
void sparseVertexFieldType(...);

void meshType(...);

template <typename Tag, typename T>
using cell_field_t = decltype(cellFieldType<T>(Tag{}));
template <typename Tag, typename T>
using edge_field_t = decltype(edgeFieldType<T>(Tag{}));
template <typename Tag, typename T>
using vertex_field_t = decltype(vertexFieldType<T>(Tag{}));

template <typename Tag, typename T>
using sparse_cell_field_t = decltype(sparseCellFieldType<T>(Tag{}));
template <typename Tag, typename T>
using sparse_edge_field_t = decltype(sparseEdgeFieldType<T>(Tag{}));
template <typename Tag, typename T>
using sparse_vertex_field_t = decltype(sparseVertexFieldType<T>(Tag{}));

template <typename Tag>
using mesh_t = decltype(meshType(Tag{}));

// generic deref, specialize if needed
template <typename Tag, typename LocationType>
auto deref(Tag, LocationType const& l) -> LocationType const& {
  return l;
}
} // namespace dawn
