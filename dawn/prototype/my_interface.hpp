#pragma once

#include "grid.hpp"

namespace MyInterface {

using Mesh = lib_lukas::Grid;
using Cell = lib_lukas::Face;
using Edge = lib_lukas::Edge;
using Node = lib_lukas::Vertex;

template <typename T>
using CellField = lib_lukas::FaceData<T>;
template <typename T>
using EdgeField = lib_lukas::EdgeData<T>;
template <typename T>
using NodeField = lib_lukas::VertexData<T>;

decltype(auto) getTriangles(Mesh const& m) { return m.faces(); }
decltype(auto) getEdges(Mesh const& m) { return m.edges(); }
decltype(auto) getVertices(Mesh const& m) { return m.vertices(); }

decltype(auto) cellNeighboursOfCell(Mesh const&, Face const& f) { return f.faces(); }
decltype(auto) nodeNeighboursOfCell(Mesh const&, Face const& f) { return f.vertices(); }
decltype(auto) edgeNeighboursOfCell(Mesh const&, Face const& f) { return f.edges(); }

decltype(auto) cellNeighboursOfEdge(Mesh const&, Edge const& e) { return e.faces(); }
decltype(auto) nodeNeighboursOfEdge(Mesh const&, Edge const& e) { return e.vertices(); }
// decltype(auto) edgeNeighboursOfEdge(Mesh const&, Edge const& e) { return e.edges(); }    //does
// this ever make sense?

decltype(auto) cellNeighboursOfNode(Mesh const&, Node const& n) { return n.faces(); }
decltype(auto) nodeNeighboursOfNode(Mesh const&, Node const& n) { return n.vertices(); }
decltype(auto) edgeNeighboursOfNode(Mesh const&, Node const& n) { return n.edges(); }

template <typename Objs, typename Init, typename Op>
auto reduce(Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

} // namespace MyInterface
