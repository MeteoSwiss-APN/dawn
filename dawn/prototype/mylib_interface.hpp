#pragma once

#include "mylib.hpp"

namespace mylibInterface {

struct mylibTag {};

template <typename T>
mylib::FaceData<T> fieldType(mylibTag);
mylib::Grid meshType(mylibTag);

decltype(auto) getCells(mylibTag, mylib::Grid const& m) { return m.faces(); }
decltype(auto) getEdges(mylibTag, mylib::Grid const& m) { return m.edges(); }
decltype(auto) getVertices(mylibTag, mylib::Grid const& m) { return m.vertices(); }

template <typename Init, typename Op>
auto reduceCellToCell(mylibTag, mylib::Grid const& grid, mylib::Face const& f, Init init, Op&& op) {
  for(auto&& obj : f.faces())
    op(init, *obj);
  return init;
}

} // namespace mylibInterface
