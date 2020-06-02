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

#include "atlas/mesh.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <set>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "driver-includes/unstructured_interface.hpp"

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
  T* data() { return atlas_field_.data(); }
  const T* data() const { return atlas_field_.data(); }
  int numElements() const { return atlas_field_.shape(0) * atlas_field_.shape(1); }

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
  T* data() { return sparse_dimension_.data(); }
  const T* data() const { return sparse_dimension_.data(); }
  int numElements() const {
    return sparse_dimension_.shape(0) * sparse_dimension_.shape(1) * sparse_dimension_.shape(2);
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

inline auto getCells(atlasTag, atlas::Mesh const& m) {
  return utility::irange(0, m.cells().size());
}
inline auto getEdges(atlasTag, atlas::Mesh const& m) {
  return utility::irange(0, m.edges().size());
}
inline auto getVertices(atlasTag, atlas::Mesh const& m) {
  return utility::irange(0, m.nodes().size());
}

inline std::vector<int> getNeighs(const atlas::Mesh::HybridElements::Connectivity& conn, int idx) {
  std::vector<int> neighs;
  for(int n = 0; n < conn.cols(idx); ++n) {
    int nbhIdx = conn(idx, n);
    if(nbhIdx != conn.missing_value()) {
      neighs.emplace_back(nbhIdx);
    }
  }
  return neighs;
}

inline std::vector<int> getNeighs(const atlas::mesh::Nodes::Connectivity& conn, int idx) {
  std::vector<int> neighs;
  for(int n = 0; n < conn.cols(idx); ++n) {
    int nbhIdx = conn(idx, n);
    if(nbhIdx != conn.missing_value()) {
      neighs.emplace_back(nbhIdx);
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
namespace {
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
} // namespace

template <typename T>
struct NotDuplicateOrOrigin {
  NotDuplicateOrOrigin(){};
  NotDuplicateOrOrigin(T origin) : origin_(origin) { compOrigin = true; };
  bool operator()(const T& element) {
    if(compOrigin && element == origin_) {
      return false;
    }
    return s_.insert(element).second; // true if s_.insert(element);
  }

private:
  std::set<T> s_;
  // optional only available in C++17, but we compile generated code with C++11
  bool compOrigin = false;
  T origin_;
};

// entry point, kicks off the recursive function above if required
inline std::vector<int> getNeighbors(atlasTag, atlas::Mesh const& mesh,
                                     std::vector<dawn::LocationType> chain, int idx) {

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

  // we want to exclude the original element from the neighborhood obtained we can compare by
  // the id, but only if targetType = startOfChain, since ids may be duplicated amongst different
  // element types; e.g. there may be a vertex and an edge with the same id.
  NotDuplicateOrOrigin<int> pred;
  if(chain.front() == chain.back()) {
    pred = NotDuplicateOrOrigin<int>(idx);
  } else {
    pred = NotDuplicateOrOrigin<int>();
  }

  // start recursion
  getNeighborsImpl(nbhTables, chain, targetType, {idx}, result);

  std::vector<int> resultUnique;
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
  for(auto&& objIdx : getNeighbors(atlasTag{}, m, chain, idx))
    op(init, objIdx, weights[i++]);
  return init;
}

//===------------------------------------------------------------------------------------------===//
// unweighted version
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op>
auto reduce(atlasTag, atlas::Mesh const& m, int idx, Init init,
            std::vector<dawn::LocationType> chain, Op&& op) {
  for(auto&& objIdx : getNeighbors(atlasTag{}, m, chain, idx))
    op(init, objIdx);
  return init;
}

} // namespace atlasInterface
