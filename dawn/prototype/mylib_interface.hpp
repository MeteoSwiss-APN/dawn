#pragma once

#include "mylib.hpp"

namespace mylibInterface {

struct mylibTag {};

mylib::Grid meshType(mylibTag);
template <typename T>
using CellField = mylib::FaceData<T>;
template <typename T>
using EdgeField = mylib::EdgeData<T>;
template <typename T>
using NodeField = mylib::VertexData<T>;

using Mesh = mylib::Grid;

decltype(auto) getCells(mylibTag, mylib::Grid const& m) { return m.faces(); }
decltype(auto) getEdges(mylibTag, mylib::Grid const& m) { return m.edges(); }
decltype(auto) getVertices(mylibTag, mylib::Grid const& m) { return m.vertices(); }

// template <typename Objs, typename Init, typename Op>
// auto reduce(Objs&& objs, Init init, Op&& op) {
//   for(auto&& obj : objs)
//     op(init, *obj);
//   return init;
// }

template <typename Init, typename Op>
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op) {
  for(auto&& obj : f.faces())
    op(init, *obj);
  return init;
}

template <typename Init, typename Op>
auto reduceEdgeToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op) {
  for(auto&& obj : f.edges())
    op(init, *obj);
  return init;
}

template <typename Init, typename Op>
auto reduceVertexToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init,
                        Op&& op) {
  for(auto&& obj : f.vertices())
    op(init, *obj);
  return init;
}

} // namespace mylibInterface
