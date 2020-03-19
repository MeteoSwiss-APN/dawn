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
#include <functional>
#include <iterator>
#include <set>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "../driver-includes/unstructured_interface.hpp"

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

int indexType(atlasTag);

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
  return getNeighs(m.cells().edge_connectivity(), idx);
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

// neighbor tables, adressable by two location types (from -> to)
typedef std::tuple<dawn::LocationType, dawn::LocationType> key_t;

struct key_hash : public std::unary_function<key_t, std::size_t> {
  std::size_t operator()(const key_t& k) const {
    return size_t(std::get<0>(k)) ^ size_t(std::get<1>(k));
  }
};

// recursive function collecting neighbors succesively
void getNeighborsImpl(
    const std::unordered_map<key_t, std::function<std::vector<int>(int)>, key_hash>& nbhTables,
    std::vector<dawn::LocationType>& chain, dawn::LocationType targetType, std::vector<int> front,
    std::list<int>& result) {
  assert(chain.size() >= 2);
  dawn::LocationType from = chain.back();
  chain.pop_back();
  dawn::LocationType to = chain.back();

  bool isNeighborOfTarget = nbhTables.count({from, targetType});

  std::vector<int> newFront;
  for(auto idx : front) {
    // Build up new front for next recursive call
    auto nextElems = nbhTables.at({from, to})(idx);
    newFront.insert(std::end(newFront), std::begin(nextElems), std::end(nextElems));

    if(isNeighborOfTarget) {
      const auto& targetElems = nbhTables.at({from, targetType})(idx);
      // Add to result set the neighbors (of target type) of current (idx)
      std::copy(targetElems.begin(), targetElems.end(), std::inserter(result, result.end()));
    }
  }

  if(chain.size() >= 2) {
    getNeighborsImpl(nbhTables, chain, targetType, newFront, result);
  }
}

template <typename T>
struct NotDuplicate {
  bool operator()(const T& element) {
    return s_.insert(element).second; // true if s_.insert(element);
  }

private:
  std::set<T> s_;
};

// entry point, kicks off the recursive function above if required
std::vector<int> getNeighbors(atlas::Mesh const& mesh, std::vector<dawn::LocationType> chain,
                              int idx) {

  // target type is at the end of the chain (we collect all neighbors of this type "along" the
  // chain)
  dawn::LocationType targetType = chain.back();

  // lets revert s.t. we can use the standard std::vector interface (pop_back() and back())
  std::reverse(std::begin(chain), std::end(chain));

  auto cellsFromEdge = [&](int edgeIdx) -> std::vector<int> {
    return getNeighs(mesh.edges().cell_connectivity(), edgeIdx);
  };
  auto nodesFromEdge = [&](int edgeIdx) -> std::vector<int> {
    return getNeighs(mesh.edges().node_connectivity(), edgeIdx);
  };

  auto cellsFromNode = [&](int nodeIdx) -> std::vector<int> {
    return getNeighs(mesh.nodes().cell_connectivity(), nodeIdx);
  };
  auto edgesFromNode = [&](int nodeIdx) -> std::vector<int> {
    return getNeighs(mesh.nodes().edge_connectivity(), nodeIdx);
  };

  auto nodesFromCell = [&](int nodeIdx) -> std::vector<int> {
    return getNeighs(mesh.cells().node_connectivity(), nodeIdx);
  };
  auto edgesFromCell = [&](int nodeIdx) -> std::vector<int> {
    return getNeighs(mesh.cells().edge_connectivity(), nodeIdx);
  };

  // convenience wrapper for all neighbor tables
  std::unordered_map<key_t, std::function<std::vector<int>(int)>, key_hash> nbhTables;
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Edges),
                    edgesFromCell);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Vertices),
                    nodesFromCell);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Cells),
                    cellsFromEdge);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Vertices),
                    nodesFromEdge);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Cells),
                    cellsFromNode);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Edges),
                    edgesFromNode);

  // result set
  std::list<int> result;

  // start recursion
  getNeighborsImpl(nbhTables, chain, targetType, {idx}, result);

  std::vector<int> resultUnique;
  NotDuplicate<int> pred;
  std::copy_if(result.begin(), result.end(), std::back_inserter(resultUnique), std::ref(pred));
  return resultUnique;
}

//===------------------------------------------------------------------------------------------===//
// weighted version
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op, typename WeightT>
auto reduce(atlasTag, atlas::Mesh const& m, int idx, Init init,
            std::vector<dawn::LocationType> chain, Op&& op, std::vector<WeightT>&& weights) {
  static_assert(std::is_arithmetic<WeightT>::value, "weights need to be of arithmetic type!\n");
  int i = 0;
  for(auto&& objIdx : getNeighbors(m, chain, idx))
    op(init, objIdx, weights[i++]);
  return init;
}

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
auto reduce(atlasTag, atlas::Mesh const& m, int idx, Init init,
            std::vector<dawn::LocationType> chain, Op&& op) {
  for(auto&& objIdx : getNeighbors(m, chain, idx))
    op(init, objIdx);
  return init;
}

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
