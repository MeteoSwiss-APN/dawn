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

#include "../driver-includes/unstructured_interface.hpp"
#include "../toylib/toylib.hpp"
#include <functional>
#include <set>
#include <unordered_map>

namespace toylibInterface {

struct toylibTag {};

toylib::Grid meshType(toylibTag);
template <typename T>
toylib::FaceData<T> cellFieldType(toylibTag);
template <typename T>
toylib::EdgeData<T> edgeFieldType(toylibTag);
template <typename T>
toylib::VertexData<T> vertexFieldType(toylibTag);

template <typename T>
toylib::SparseEdgeData<T> sparseEdgeFieldType(toylibTag);
template <typename T>
toylib::SparseFaceData<T> sparseCellFieldType(toylibTag);
template <typename T>
toylib::SparseVertexData<T> sparseVertexFieldType(toylibTag);

using Mesh = toylib::Grid;

inline decltype(auto) getCells(toylibTag, toylib::Grid const& m) { return m.faces(); }
inline decltype(auto) getEdges(toylibTag, toylib::Grid const& m) { return m.edges(); }
inline decltype(auto) getVertices(toylibTag, toylib::Grid const& m) { return m.vertices(); }

// Specialized to deref the reference_wrapper
inline toylib::Edge const& deref(toylibTag, std::reference_wrapper<toylib::Edge> const& e) {
  return e; // implicit conversion
}

inline toylib::Edge const& deref(toylibTag, std::reference_wrapper<const toylib::Edge> const& e) {
  return e; // implicit conversion
}

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
inline void getNeighborsImpl(
    const std::unordered_map<
        key_t,
        std::function<std::vector<const toylib::ToylibElement*>(const toylib::ToylibElement*)>,
        key_hash, key_equal>& nbhTables,
    std::vector<dawn::LocationType>& chain, dawn::LocationType targetType,
    std::vector<const toylib::ToylibElement*> front,
    std::set<const toylib::ToylibElement*>& result) {
  dawn::LocationType from = chain.back();
  chain.pop_back();
  dawn::LocationType to = chain.back();

  std::vector<const toylib::ToylibElement*> newFront;
  for(auto elem : front) {
    auto nextElems = nbhTables.at({from, to})(elem);
    newFront.insert(std::end(newFront), std::begin(nextElems), std::end(nextElems));
  }

  if(to == targetType) {
    std::copy(newFront.begin(), newFront.end(), std::inserter(result, result.end()));
  }

  if(chain.size() >= 2) {
    getNeighborsImpl(nbhTables, chain, targetType, newFront, result);
  }
}

inline std::vector<const toylib::ToylibElement*> getNeighbors(const toylib::Grid& mesh,
                                                              std::vector<dawn::LocationType> chain,
                                                              toylib::ToylibElement* elem) {
  // target type is at the end of the chain (we collect all neighbors of this type "along" the
  // chain)
  dawn::LocationType targetType = chain.back();

  // lets revert s.t. we can use the standard std::vector interface (pop_back() and back())
  std::reverse(std::begin(chain), std::end(chain));

  // consume first element in chain (where we currently are, "from")
  dawn::LocationType from = chain.back();
  chain.pop_back();

  // look at next element
  dawn::LocationType to = chain.back();

  auto cellsFromEdge =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Edge* edge = static_cast<const toylib::Edge*>(elem);
    auto faces = edge->faces();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(faces.begin(), faces.end(), std::back_inserter(ret),
                   [](const toylib::Face* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };
  auto verticesFromEdge =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Edge* edge = static_cast<const toylib::Edge*>(elem);
    auto vertices = edge->vertices();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(vertices.begin(), vertices.end(), std::back_inserter(ret),
                   [](const toylib::Vertex* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };

  auto verticesFromCell =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Face* face = static_cast<const toylib::Face*>(elem);
    auto vertices = face->vertices();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(vertices.begin(), vertices.end(), std::back_inserter(ret),
                   [](const toylib::Vertex* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };
  auto edgesFromCell =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Face* face = static_cast<const toylib::Face*>(elem);
    auto edges = face->edges();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(edges.begin(), edges.end(), std::back_inserter(ret),
                   [](const toylib::Edge* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };

  auto cellsFromVertex =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Vertex* vertex = static_cast<const toylib::Vertex*>(elem);
    auto faces = vertex->faces();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(faces.begin(), faces.end(), std::back_inserter(ret),
                   [](const toylib::Face* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };
  auto edgesFromVertex =
      [](const toylib::ToylibElement* elem) -> std::vector<const toylib::ToylibElement*> {
    const toylib::Vertex* vertex = static_cast<const toylib::Vertex*>(elem);
    auto edges = vertex->edges();
    std::vector<const toylib::ToylibElement*> ret;
    std::transform(edges.begin(), edges.end(), std::back_inserter(ret),
                   [](const toylib::Edge* in) -> const toylib::ToylibElement* {
                     return static_cast<const toylib::ToylibElement*>(in);
                   });
    return ret;
  };

  std::unordered_map<
      key_t, std::function<std::vector<const toylib::ToylibElement*>(const toylib::ToylibElement*)>,
      key_hash, key_equal>
      nbhTables;

  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Cells),
                    cellsFromEdge);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Edges, dawn::LocationType::Vertices),
                    verticesFromEdge);

  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Vertices),
                    verticesFromCell);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Cells, dawn::LocationType::Edges),
                    edgesFromCell);

  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Cells),
                    cellsFromVertex);
  nbhTables.emplace(std::make_tuple(dawn::LocationType::Vertices, dawn::LocationType::Edges),
                    edgesFromVertex);

  // update the current from (the neighbors we can reach from the current index)
  std::vector<const toylib::ToylibElement*> front = nbhTables.at({from, to})(elem);

  // result set
  std::set<const toylib::ToylibElement*> result;

  // if next element is of target type we collect the current front into the result
  if(to == targetType) {
    std::copy(front.begin(), front.end(), std::inserter(result, result.end()));
  }

  // if there are two or more elements in the chain remaining, we need to recursively keep
  // collecting neighbors
  if(chain.size() >= 2) {
    getNeighborsImpl(nbhTables, chain, targetType, front, result);
  }

  return std::vector<const toylib::ToylibElement*>(result.begin(), result.end());
}

//===------------------------------------------------------------------------------------------===//
// unweighted versions
//===------------------------------------------------------------------------------------------===//

template <typename Objs, typename Init, typename Op>
auto reduce(Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

template <typename Init, typename Op>
auto reduceCellToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                      Op&& op) {
  return reduce(f.faces(), init, op);
}
template <typename Init, typename Op>
auto reduceEdgeToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                      Op&& op) {
  return reduce(f.edges(), init, op);
}
template <typename Init, typename Op>
auto reduceVertexToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                        Op&& op) {
  return reduce(f.vertices(), init, op);
}
template <typename Init, typename Op>
auto reduceCellToEdge(toylibTag, toylib::Grid const& grid, toylib::Edge const& e, Init init,
                      Op&& op) {
  return reduce(e.faces(), init, op);
}
template <typename Init, typename Op>
auto reduceVertexToEdge(toylibTag, toylib::Grid const& grid, toylib::Edge const& e, Init init,
                        Op&& op) {
  return reduce(e.vertices(), init, op);
}
template <typename Init, typename Op>
auto reduceCellToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                        Op&& op) {
  return reduce(v.faces(), init, op);
}
template <typename Init, typename Op>
auto reduceEdgeToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                        Op&& op) {
  return reduce(v.edges(), init, op);
}
template <typename Init, typename Op>
auto reduceVertexToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                          Op&& op) {
  return reduce(v.vertices(), init, op);
}

//===------------------------------------------------------------------------------------------===//
// weighted versions
//===------------------------------------------------------------------------------------------===//

template <typename Objs, typename Init, typename Op, typename Weight>
auto reduce(Objs&& objs, Init init, Op&& op, std::vector<Weight>&& weights) {
  int i = 0;
  for(auto&& obj : objs)
    op(init, *obj, weights[i++]);
  return init;
}

template <typename Init, typename Op, typename Weight>
auto reduceCellToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                      Op&& op, std::vector<Weight>&& weights) {
  return reduce(f.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceEdgeToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                      Op&& op, std::vector<Weight>&& weights) {
  return reduce(f.edges(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToCell(toylibTag, toylib::Grid const& grid, toylib::Face const& f, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(f.vertices(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceCellToEdge(toylibTag, toylib::Grid const& grid, toylib::Edge const& e, Init init,
                      Op&& op, std::vector<Weight>&& weights) {
  return reduce(e.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToEdge(toylibTag, toylib::Grid const& grid, toylib::Edge const& e, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(e.vertices(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceCellToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceEdgeToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.edges(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToVertex(toylibTag, toylib::Grid const& grid, toylib::Vertex const& v, Init init,
                          Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.vertices(), init, op, std::move(weights));
}

} // namespace toylibInterface