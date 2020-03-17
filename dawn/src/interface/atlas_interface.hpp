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

// NOT working with std::variant
//  - connectivity tables are returned as refs
//  - refs are not allowed as variants
//  - variants need to be default constructible, reference_wrappers are not
// class ConnInterface {
// private:
//   std::variant<atlas::mesh::HybridElements::Connectivity*, atlas::mesh::Nodes::Connectivity*>
//       nbhTable_;

// public:
//   ConnInterface(atlas::mesh::HybridElements::Connectivity* nbhTable) : nbhTable_(nbhTable){};
//   ConnInterface(atlas::mesh::Nodes::Connectivity* nbhTable) : nbhTable_(nbhTable){};

//   size_t cols(size_t idx) {
//     if(auto pval = std::get_if<atlas::mesh::HybridElements::Connectivity*>(&nbhTable_)) {
//       return (*pval)->cols(idx);
//     } else if(auto pval = std::get_if<atlas::mesh::Nodes::Connectivity*>(&nbhTable_)) {
//       return (*pval)->cols(idx);
//     } else {
//       assert(false);
//     }
//   }
//   size_t operator()(size_t elem_idx, size_t nbh_idx) {
//     if(auto pval = std::get_if<atlas::mesh::HybridElements::Connectivity*>(&nbhTable_)) {
//       return (**pval)(elem_idx, nbh_idx);
//     } else if(auto pval = std::get_if<atlas::mesh::Nodes::Connectivity*>(&nbhTable_)) {
//       return (**pval)(elem_idx, nbh_idx);
//     } else {
//       assert(false);
//     }
//   }
// };

class ConnInterface {
private:
  enum class ConnType { Nodes, Hybrid };
  ConnType connType_;
  // can't store a reference dirctly because reference members need all be set upon creaton of the
  // object
  // can't use a reference wrapper directly because they are not default constructible
  // pointers to references are illegal
  // so here goes optional reference wrappers, I guess?
  std::optional<std::reference_wrapper<const atlas::mesh::Nodes::Connectivity>> nodesTable_;
  std::optional<std::reference_wrapper<const atlas::mesh::HybridElements::Connectivity>>
      hybridTable_;

public:
  ConnInterface(const atlas::mesh::HybridElements::Connectivity& nbhTable)
      : hybridTable_(nbhTable) {
    connType_ = ConnType::Hybrid;
  };
  ConnInterface(const atlas::mesh::Nodes::Connectivity& nbhTable) : nodesTable_(nbhTable) {
    connType_ = ConnType::Nodes;
  };

  size_t cols(size_t idx) {
    switch(connType_) {
    case ConnType::Nodes:
      return (*nodesTable_).get().cols(idx);
    case ConnType::Hybrid:
      return (*hybridTable_).get().cols(idx);
    }
    throw std::runtime_error("unreachable");
  }

  size_t operator()(size_t elem_idx, size_t nbh_idx) {
    switch(connType_) {
    case ConnType::Nodes:
      return (*nodesTable_).get()(elem_idx, nbh_idx);
    case ConnType::Hybrid:
      return (*hybridTable_).get()(elem_idx, nbh_idx);
    }
    throw std::runtime_error("unreachable");
  }
};

// neighbor tables, adressable by two location types (from -> to)
typedef std::tuple<dawn::LocationType, dawn::LocationType> key_t;

struct key_hash : public std::unary_function<key_t, std::size_t> {
  std::size_t operator()(const key_t& k) const {
    return size_t(std::get<0>(k)) ^ size_t(std::get<1>(k));
  }
};

struct key_equal : public std::binary_function<key_t, key_t, bool> {
  bool operator()(const key_t& v0, const key_t& v1) const {
    return (std::get<0>(v0) == std::get<0>(v1) && std::get<1>(v0) == std::get<1>(v1));
  }
};

// recursive function collecting neighbors succesively
void getNeighborsImpl(
    const std::unordered_map<key_t, ConnInterface, key_hash, key_equal>& nbhTables,
    std::vector<dawn::LocationType>& chain, dawn::LocationType targetType, std::vector<int> front,
    std::set<int>& result) {
  dawn::LocationType from = chain.back();
  chain.pop_back();
  dawn::LocationType to = chain.back();

  auto table = nbhTables.at({from, to});

  std::vector<int> newFront;
  for(auto idx : front) {
    for(int nbhIdx = 0; nbhIdx < table.cols(idx); nbhIdx++) {
      newFront.push_back(table(idx, nbhIdx));
    }
  }

  if(to == targetType) {
    std::copy(newFront.begin(), newFront.end(), std::inserter(result, result.end()));
  }

  if(chain.size() >= 2) {
    getNeighborsImpl(nbhTables, chain, targetType, newFront, result);
  }
}

// entry point, kicks off the recursive function above if required
std::vector<int> getNeighbors(atlas::Mesh const& mesh, std::vector<dawn::LocationType> chain,
                              int idx) {

  // target type is at the end of the chain (we collect all neighbors of this type "along" the
  // chain)
  dawn::LocationType targetType = chain.back();

  // lets revert s.t. we can use the standard std::vector interface (pop_back() and back())
  std::reverse(std::begin(chain), std::end(chain));

  // convenience wrapper for all neighbor tables
  std::unordered_map<key_t, ConnInterface, key_hash, key_equal> nbhTables;
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Edges),
                    ConnInterface(mesh.cells().edge_connectivity()));
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Vertices),
                    ConnInterface(mesh.cells().node_connectivity()));
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Vertices),
                    ConnInterface(mesh.edges().cell_connectivity()));
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Vertices),
                    ConnInterface(mesh.edges().node_connectivity()));
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Cells),
                    ConnInterface(mesh.nodes().cell_connectivity()));
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Edges),
                    ConnInterface(mesh.nodes().edge_connectivity()));

  // consume first element in chain (where we currently are, "from")
  dawn::LocationType from = chain.back();
  chain.pop_back();

  // look at next element
  dawn::LocationType to = chain.back();

  // retrieve from->to nbh table
  auto table = nbhTables.at({from, to});

  // update the current from (the neighbors we can reach from the current index)
  std::vector<int> front;
  for(int nbhIdx = 0; nbhIdx < table.cols(idx); nbhIdx++) {
    front.push_back(table(idx, nbhIdx));
  }

  // result set
  std::set<int> result;

  // if next element is of target type we collect the current front into the result
  if(to == targetType) {
    std::copy(front.begin(), front.end(), std::inserter(result, result.end()));
  }

  // if there are two or more elements in the chain remaining, we need to recursively keep
  // collecting neighbors
  if(chain.size() >= 2) {
    getNeighborsImpl(nbhTables, chain, targetType, front, result);
  }

  return std::vector<int>(result.begin(), result.end());
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
