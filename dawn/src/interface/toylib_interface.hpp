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

#include <assert.h>
#include <functional>
#include <list>
#include <set>
#include <unordered_map>

namespace toylibInterface {

struct toylibTag {};

toylib::Grid meshType(toylibTag);
const toylib::ToylibElement* indexType(toylibTag);
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

inline std::vector<const toylib::ToylibElement*> getCells(toylibTag, toylib::Grid const& m) {
  std::vector<const toylib::ToylibElement*> ret;
  std::transform(m.faces().begin(), m.faces().end(), std::back_inserter(ret),
                 [](const toylib::Face& in) -> const toylib::ToylibElement* {
                   return static_cast<const toylib::ToylibElement*>(&in);
                 });
  return ret;
}
inline std::vector<const toylib::ToylibElement*> getEdges(toylibTag, toylib::Grid const& m) {
  std::vector<const toylib::ToylibElement*> ret;
  std::transform(m.edges().begin(), m.edges().end(), std::back_inserter(ret),
                 [](const toylib::Edge& in) -> const toylib::ToylibElement* {
                   return static_cast<const toylib::ToylibElement*>(&in);
                 });
  return ret;
}
inline std::vector<const toylib::ToylibElement*> getVertices(toylibTag, toylib::Grid const& m) {
  std::vector<const toylib::ToylibElement*> ret;
  std::transform(m.vertices().begin(), m.vertices().end(), std::back_inserter(ret),
                 [](const toylib::Vertex& in) -> const toylib::ToylibElement* {
                   return static_cast<const toylib::ToylibElement*>(&in);
                 });
  return ret;
}

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

// recursive function collecting neighbors succesively
inline void getNeighborsImpl(
    const std::unordered_map<
        key_t,
        std::function<std::vector<const toylib::ToylibElement*>(const toylib::ToylibElement*)>,
        key_hash>& nbhTables,
    std::vector<dawn::LocationType>& chain, dawn::LocationType targetType,
    std::vector<const toylib::ToylibElement*> front,
    std::list<const toylib::ToylibElement*>& result) {
  dawn::LocationType from = chain.back();
  chain.pop_back();
  dawn::LocationType to = chain.back();

  bool isNeighborOfTarget = nbhTables.count({from, targetType});

  std::vector<const toylib::ToylibElement*> newFront;
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
struct NotDuplicateOrOrigin {
  NotDuplicateOrOrigin(){};
  NotDuplicateOrOrigin(T origin) : origin_(origin) { compOrigin = true; };
  bool operator()(const T& element) {
    if(compOrigin && element->id() == origin_->id()) {
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

inline std::vector<const toylib::ToylibElement*> getNeighbors(const toylib::Grid& mesh,
                                                              std::vector<dawn::LocationType> chain,
                                                              const toylib::ToylibElement* elem) {
  switch(chain.front()) {
  case dawn::LocationType::Cells:
    assert(dynamic_cast<const toylib::Face*>(elem) != nullptr);
    break;
  case dawn::LocationType::Edges:
    assert(dynamic_cast<const toylib::Edge*>(elem) != nullptr);
    break;
  case dawn::LocationType::Vertices:
    assert(dynamic_cast<const toylib::Vertex*>(elem) != nullptr);
    break;
  }
  // target type is at the end of the chain (we collect all neighbors of this type "along" the
  // chain)
  dawn::LocationType targetType = chain.back();

  // lets revert s.t. we can use the standard std::vector interface (pop_back() and back())
  std::reverse(std::begin(chain), std::end(chain));

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
      key_hash>
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

  // result set
  std::list<const toylib::ToylibElement*> result;

  // we want to exclude the original element from the neighborhood obtained. We the value of elem*
  // since this might be the adress of a temporary assigned by the user. However, we can compare by
  // the id, but only if targetType = startOfChain, since ids may be duplicated amongst different
  // element types; e.g. there may be a vertex and an edge with the same id.
  NotDuplicateOrOrigin<const toylib::ToylibElement*> pred;
  if(chain.front() == chain.back()) {
    pred = NotDuplicateOrOrigin<const toylib::ToylibElement*>(elem);
  } else {
    pred = NotDuplicateOrOrigin<const toylib::ToylibElement*>();
  }

  // start recursion
  getNeighborsImpl(nbhTables, chain, targetType, {elem}, result);

  std::vector<const toylib::ToylibElement*> resultUnique;
  std::copy_if(result.begin(), result.end(), std::back_inserter(resultUnique), std::ref(pred));
  return resultUnique;
}

//===------------------------------------------------------------------------------------------===//
// unweighted version
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op>
auto reduce(toylibTag, toylib::Grid const& grid, toylib::ToylibElement const* idx, Init init,
            std::vector<dawn::LocationType> chain, Op&& op) {
  for(auto ptr : getNeighbors(grid, chain, idx)) {
    switch(chain.back()) {
    case dawn::LocationType::Cells:
      op(init, static_cast<const toylib::Face*>(ptr));
      break;
    case dawn::LocationType::Edges:
      op(init, static_cast<const toylib::Edge*>(ptr));
      break;
    case dawn::LocationType::Vertices:
      op(init, static_cast<const toylib::Vertex*>(ptr));
      break;
    }
  }

  return init;
}

//===------------------------------------------------------------------------------------------===//
// weighted version
//===------------------------------------------------------------------------------------------===//

template <typename Init, typename Op, typename Weight>
auto reduce(toylibTag, toylib::Grid const& grid, toylib::ToylibElement const* idx, Init init,
            std::vector<dawn::LocationType> chain, Op&& op, std::vector<Weight>&& weights) {
  int i = 0;
  for(auto ptr : getNeighbors(grid, chain, idx)) {
    switch(chain.back()) {
    case dawn::LocationType::Cells:
      op(init, static_cast<const toylib::Face*>(ptr), weights[i++]);
      break;
    case dawn::LocationType::Edges:
      op(init, static_cast<const toylib::Edge*>(ptr), weights[i++]);
      break;
    case dawn::LocationType::Vertices:
      op(init, static_cast<const toylib::Vertex*>(ptr), weights[i++]);
      break;
    }
  }

  return init;
}

} // namespace toylibInterface