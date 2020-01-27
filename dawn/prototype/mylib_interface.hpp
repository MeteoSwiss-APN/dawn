#ifndef DAWN_PROTOTYPE_MYLIB_INTERFACE_H_
#define DAWN_PROTOTYPE_MYLIB_INTERFACE_H_

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
mylib::SparseData<T> sparseDimensionType(mylibTag);

using Mesh = mylib::Grid;

inline decltype(auto) getCells(mylibTag, mylib::Grid const& m) { return m.faces(); }
inline decltype(auto) getEdges(mylibTag, mylib::Grid const& m) { return m.edges(); }
inline decltype(auto) getVertices(mylibTag, mylib::Grid const& m) { return m.vertices(); }

// Specialized to deref the reference_wrapper
inline mylib::Edge const& deref(mylibTag, std::reference_wrapper<mylib::Edge> const& e) {
  return e; // implicit conversion
}

//===------------------------------------------------------------------------------------------===//
// unweighted, no sparse dimensions
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
// weighted, no sparse dimensions
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

//===------------------------------------------------------------------------------------------===//
// unweighted, sparse dimensions
//===------------------------------------------------------------------------------------------===//

template <typename Objs, typename Init, typename Op, typename SparseDimT>
auto reduce(Objs&& objs, int objIdx, Init init, Op&& op, int k_level,
            const mylib::SparseData<SparseDimT>& sparseDimension) {
  int i = 0;
  for(auto&& obj : objs)
    op(init, *obj, sparseDimension(objIdx, k_level, i++));
  return init;
}

template <typename Init, typename Op, typename SparseDimT>
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                      Init init, Op&& op, const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.faces(), f.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceEdgeToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                      Init init, Op&& op, const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.edges(), f.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceVertexToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                        Init init, Op&& op, const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.vertices(), f.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceCellToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, int k_level,
                      Init init, Op&& op, mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(e.faces(), e.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceVertexToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, int k_level,
                        Init init, Op&& op, mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(e.vertices(), e.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceCellToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                        Init init, Op&& op, mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.faces(), v.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceEdgeToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                        Init init, Op&& op, mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.edges(), v.id(), init, op, k_level, sparseDimension);
}
template <typename Init, typename Op, typename SparseDimT>
auto reduceVertexToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                          Init init, Op&& op, mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.vertices(), v.id(), init, op, k_level, sparseDimension);
}

//===------------------------------------------------------------------------------------------===//
// weighted, sparse dimensions
//===------------------------------------------------------------------------------------------===//

template <typename Objs, typename Init, typename Op, typename SparseDimT, typename WeightT>
auto reduce(Objs&& objs, int objIdx, Init init, Op&& op, int k_level,
            const mylib::SparseData<SparseDimT>& sparseDimension, std::vector<WeightT>&& weights) {
  int i = 0;
  for(auto&& obj : objs) {
    op(init, *obj, sparseDimension(objIdx, k_level, i), weights[i]);
    i++;
  }
  return init;
}

template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                      Init init, Op&& op, std::vector<WeightT>&& weights,
                      const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.faces(), f.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceEdgeToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                      Init init, Op&& op, std::vector<WeightT>&& weights,
                      const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.edges(), f.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceVertexToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, int k_level,
                        Init init, Op&& op, std::vector<WeightT>&& weights,
                        const mylib::SparseData<SparseDimT>& sparseDimension) {
  return reduce(f.vertices(), f.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceCellToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, int k_level,
                      Init init, Op&& op, std::vector<WeightT>&& weights,
                      mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(e.faces(), e.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceVertexToEdge(mylibTag, mylib::Grid const& grid, mylib::Edge const& e, int k_level,
                        Init init, Op&& op, std::vector<WeightT>&& weights,
                        mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(e.vertices(), e.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceCellToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                        Init init, Op&& op, std::vector<WeightT>&& weights,
                        mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.faces(), v.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceEdgeToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                        Init init, Op&& op, std::vector<WeightT>&& weights,
                        mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.edges(), v.id(), init, op, k_level, sparseDimension, std::move(weights));
}
template <typename Init, typename Op, typename WeightT, typename SparseDimT>
auto reduceVertexToVertex(mylibTag, mylib::Grid const& grid, mylib::Vertex const& v, int k_level,
                          Init init, Op&& op, std::vector<WeightT>&& weights,
                          mylib::SparseData<SparseDimT>&& sparseDimension) {
  return reduce(v.vertices(), v.id(), init, op, k_level, sparseDimension, std::move(weights));
}

} // namespace mylibInterface

#endif // DAWN_PROTOTYPE_MYLIB_INTERFACE_H_
