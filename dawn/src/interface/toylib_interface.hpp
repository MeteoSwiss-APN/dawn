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