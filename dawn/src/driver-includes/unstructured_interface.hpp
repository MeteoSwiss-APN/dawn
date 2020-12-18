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
#include "extent.hpp"

#include <vector>

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

template <typename T>
void verticalFieldType(...);

void meshType(...);

void indexType(...);

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

template <typename Tag, typename T>
using vertical_field_t = decltype(verticalFieldType<T>(Tag{}));

template <typename Tag>
using nbh_table_index_t = decltype(indexType(Tag{}));

template <typename Tag>
using mesh_t = decltype(meshType(Tag{}));

// TODO there is currently no convenient way to share code internal to dawn with driver code
//      leading to reproduciton of some dawn internatls like the following enum and typedef
enum class LocationType { Cells = 0, Edges, Vertices };
using UnstructuredIterationSpace = std::tuple<std::vector<LocationType>, bool>;

// generic deref, specialize if needed
template <typename Tag, typename LocationType>
auto deref(Tag, LocationType const& l) -> LocationType const& {
  return l;
}

template <typename Tag, typename LocationType>
auto deref(Tag, LocationType const* l) -> LocationType const* {
  return l;
}

} // namespace dawn
