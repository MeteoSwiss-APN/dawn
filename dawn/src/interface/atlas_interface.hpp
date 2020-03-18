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

#ifndef DAWN_INTERFACE_ATLAS_INTERFACE_H_
#define DAWN_INTERFACE_ATLAS_INTERFACE_H_

#include "atlas/mesh.h"
#include <algorithm>
#include <cassert>
#include <iterator>

namespace utility {
namespace impl_ {
template <typename Integer>
class irange_ {
public:
  class iterator {
  public:
    Integer operator*() const { return i_; }
    const iterator& operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }

    bool operator==(const iterator& other) const { return i_ == other.i_; }
    bool operator!=(const iterator& other) const { return i_ != other.i_; }

    iterator(Integer start) : i_(start) {}

  private:
    Integer i_;
  };

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }
  irange_(Integer begin, Integer end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};
} // namespace impl_
template <typename Integer>
impl_::irange_<Integer> irange(Integer from, Integer to) {
  return {from, to};
}
} // namespace utility

namespace atlasInterface {

struct atlasTag {};

template <typename T>
class Field {
public:
  T const& operator()(int f, int k) const { return atlas_field_(f, k); }
  T& operator()(int f, int k) { return atlas_field_(f, k); }

  Field(atlas::array::ArrayView<T, 2> const& atlas_field) : atlas_field_(atlas_field) {}

private:
  atlas::array::ArrayView<T, 2> atlas_field_;
};

template <typename T>
Field<T> cellFieldType(atlasTag);
template <typename T>
Field<T> edgeFieldType(atlasTag);
template <typename T>
Field<T> vertexFieldType(atlasTag);

template <typename T>
class SparseDimension {
public:
  T const& operator()(int elem_idx, int sparse_dim_idx, int level) const {
    return sparse_dimension_(elem_idx, level, sparse_dim_idx);
  }
  T& operator()(int elem_idx, int sparse_dim_idx, int level) {
    return sparse_dimension_(elem_idx, level, sparse_dim_idx);
  }

  SparseDimension(atlas::array::ArrayView<T, 3> const& sparse_dimension)
      : sparse_dimension_(sparse_dimension) {}

private:
  atlas::array::ArrayView<T, 3> sparse_dimension_;
};

template <typename T>
SparseDimension<T> sparseCellFieldType(atlasTag);
template <typename T>
SparseDimension<T> sparseEdgeFieldType(atlasTag);
template <typename T>
SparseDimension<T> sparseVertexFieldType(atlasTag);

atlas::Mesh meshType(atlasTag);

auto getCells(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.cells().size()); }
auto getEdges(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.edges().size()); }
auto getVertices(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.nodes().size()); }

std::vector<int> getNeighs(const atlas::Mesh::HybridElements::Connectivity& conn, int idx) {
  std::vector<int> neighs;
  for(int n = 0; n < conn.cols(idx); ++n) {
    neighs.emplace_back(conn(idx, n));
  }
  return neighs;
}

std::vector<int> getNeighs(const atlas::mesh::Nodes::Connectivity& conn, int idx) {
  std::vector<int> neighs;
  for(int n = 0; n < conn.cols(idx); ++n) {
    neighs.emplace_back(conn(idx, n));
  }
  return neighs;
}

std::vector<int> const cellNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  const auto& conn = m.cells().edge_connectivity();
  auto neighs = std::vector<int>{};
  for(int n = 0; n < conn.cols(idx); ++n) {
    int initialEdge = conn(idx, n);
    for(int c1 = 0; c1 < m.cells().size(); ++c1) {
      for(int n1 = 0; n1 < conn.cols(c1); ++n1) {
        int compareEdge = conn(c1, n1);
        if(initialEdge == compareEdge && c1 != idx) {
          neighs.emplace_back(c1);
        }
      }
    }
  }
  return neighs;
}

std::vector<int> const edgeNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  return getNeighs(m.cells().edge_connectivity(), idx);
}

std::vector<int> const nodeNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  return getNeighs(m.cells().node_connectivity(), idx);
}

std::vector<int> const cellNeighboursOfEdge(atlas::Mesh const& m, int const& idx) {
  auto neighs = getNeighs(m.edges().cell_connectivity(), idx);
  assert(neighs.size() == 2);
  return neighs;
}

std::vector<int> const nodeNeighboursOfEdge(atlas::Mesh const& m, int const& idx) {
  auto neighs = getNeighs(m.edges().node_connectivity(), idx);
  assert(neighs.size() == 2);
  return neighs;
}

std::vector<int> const cellNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  return getNeighs(m.nodes().cell_connectivity(), idx);
}

std::vector<int> const edgeNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  return getNeighs(m.nodes().edge_connectivity(), idx);
}

std::vector<int> const nodeNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  const auto& conn_nodes_to_edge = m.nodes().edge_connectivity();
  auto neighs = std::vector<int>{};
  for(int ne = 0; ne < conn_nodes_to_edge.cols(idx); ++ne) {
    int nbh_edge_idx = conn_nodes_to_edge(idx, ne);
    const auto& conn_edge_to_nodes = m.edges().node_connectivity();
    for(int nn = 0; nn < conn_edge_to_nodes.cols(nbh_edge_idx); ++nn) {
      int nbhNode = conn_edge_to_nodes(idx, nn);
      if(nbhNode != idx) {
        neighs.emplace_back(nbhNode);
      }
    }
  }
  return neighs;
}

//===------------------------------------------------------------------------------------------===//
// weighted version
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op, typename WeightT>
auto reduceCellToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                      std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : cellNeighboursOfCell(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}
template <typename Init, typename Op, typename WeightT>
auto reduceEdgeToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                      std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : edgeNeighboursOfCell(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}
template <typename Init, typename Op, typename WeightT>
auto reduceVertexToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                        std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : nodeNeighboursOfCell(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}

template <typename Init, typename Op, typename WeightT>
auto reduceCellToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                      std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : cellNeighboursOfEdge(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}
template <typename Init, typename Op, typename WeightT>
auto reduceVertexToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                        std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : nodeNeighboursOfEdge(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}

template <typename Init, typename Op, typename WeightT>
auto reduceCellToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                        std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : cellNeighboursOfNode(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}
template <typename Init, typename Op, typename WeightT>
auto reduceEdgeToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                        std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : edgeNeighboursOfNode(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}
template <typename Init, typename Op, typename WeightT>
auto reduceVertexToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op,
                          std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : nodeNeighboursOfNode(m, idx))
    op(init, objIdx, weights[i++]);
  return init;
}

//===------------------------------------------------------------------------------------------===//
// unweighted versions
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op>
auto reduceCellToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : cellNeighboursOfCell(m, idx))
    op(init, objIdx);
  return init;
}
template <typename Init, typename Op>
auto reduceEdgeToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : edgeNeighboursOfCell(m, idx))
    op(init, objIdx);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : nodeNeighboursOfCell(m, idx))
    op(init, objIdx);
  return init;
}

template <typename Init, typename Op>
auto reduceCellToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : cellNeighboursOfEdge(m, idx))
    op(init, objIdx);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : nodeNeighboursOfEdge(m, idx))
    op(init, objIdx);
  return init;
}

template <typename Init, typename Op>
auto reduceCellToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : cellNeighboursOfNode(m, idx))
    op(init, objIdx);
  return init;
}
template <typename Init, typename Op>
auto reduceEdgeToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : edgeNeighboursOfNode(m, idx))
    op(init, objIdx);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& objIdx : nodeNeighboursOfNode(m, idx))
    op(init, objIdx);
  return init;
}

} // namespace atlasInterface
#endif
