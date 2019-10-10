#ifndef DAWN_PROTOTYPE_MYLIB_INTERFACE_H_
#define DAWN_PROTOTYPE_MYLIB_INTERFACE_H_

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

template <typename Init, typename Op>
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op) {
  return reduce(f.faces(), init, op);
}
template <typename Init, typename Op>
auto reduceEdgeToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op) {
  return reduce(f.edges(), init, op);
}
template <typename Init, typename Op>
auto reduceVertexToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init,
                        Op&& op) {
  return reduce(f.vertices(), init, op);
}

template <typename Init, typename Op>
auto reduceCellToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, Init init, Op&& op) {
  return reduce(e.faces(), init, op);
}
// template <typename Init, typename Op>
// auto reduceEdgeToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, Init init, Op&&
// op) {
//   return reduce(e.edges(), init, op);
// }
template <typename Init, typename Op>
auto reduceVertexToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, Init init,
                        Op&& op) {
  return reduce(e.vertices(), init, op);
}

template <typename Init, typename Op>
auto reduceCellToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                        Op&& op) {
  return reduce(v.faces(), init, op);
}
template <typename Init, typename Op>
auto reduceEdgeToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                        Op&& op) {
  return reduce(v.edges(), init, op);
}
template <typename Init, typename Op>
auto reduceVertexToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                          Op&& op) {
  return reduce(v.vertices(), init, op);
}

template <typename Objs, typename Init, typename Op>
auto reduce(Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

} // namespace mylibInterface

#endif // DAWN_PROTOTYPE_MYLIB_INTERFACE_H_