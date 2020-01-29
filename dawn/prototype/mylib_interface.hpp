#ifndef DAWN_PROTOTYPE_MYLIB_INTERFACE_H_
#define DAWN_PROTOTYPE_MYLIB_INTERFACE_H_

#include "driver-includes/unstructured_interface.hpp"
#include "mylib.hpp"
#include <functional>

namespace mylibInterface {

struct mylibTag {};

mylib::Grid meshType(mylibTag);
template <typename T>
mylib::FaceData<T> cellFieldType(mylibTag);
template <typename T>
mylib::EdgeData<T> edgeFieldType(mylibTag);
template <typename T>
mylib::VertexData<T> vertexFieldType(mylibTag);

template <typename T>
mylib::SparseEdgeData<T> sparseEdgeFieldType(mylibTag);
template <typename T>
mylib::SparseFaceData<T> sparseCellFieldType(mylibTag);
template <typename T>
mylib::SparseVertexData<T> sparseVertexFieldType(mylibTag);

using Mesh = mylib::Grid;

inline decltype(auto) getCells(mylibTag, mylib::Grid const& m) { return m.faces(); }
inline decltype(auto) getEdges(mylibTag, mylib::Grid const& m) { return m.edges(); }
inline decltype(auto) getVertices(mylibTag, mylib::Grid const& m) { return m.vertices(); }

// Specialized to deref the reference_wrapper
inline mylib::Edge const& deref(mylibTag, std::reference_wrapper<mylib::Edge> const& e) {
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
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op,
                      std::vector<Weight>&& weights) {
  return reduce(f.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceEdgeToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op,
                      std::vector<Weight>&& weights) {
  return reduce(f.edges(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op,
                        std::vector<Weight>&& weights) {
  return reduce(f.vertices(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceCellToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, Init init, Op&& op,
                      std::vector<Weight>&& weights) {
  return reduce(e.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, Init init, Op&& op,
                        std::vector<Weight>&& weights) {
  return reduce(e.vertices(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceCellToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.faces(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceEdgeToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                        Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.edges(), init, op, std::move(weights));
}
template <typename Init, typename Op, typename Weight>
auto reduceVertexToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, Init init,
                          Op&& op, std::vector<Weight>&& weights) {
  return reduce(v.vertices(), init, op, std::move(weights));
}

} // namespace mylibInterface

#endif // DAWN_PROTOTYPE_MYLIB_INTERFACE_H_
