#pragma once

#include "grid.hpp"

namespace mylibInterface {

struct mylibTag {};

using Mesh = lib_lukas::Grid;
using Face = lib_lukas::Face;
template <typename T>
using Field = lib_lukas::FaceData<T>;

decltype(auto) getCells(mylibTag, Mesh const& m) { return m.faces(); }

decltype(auto) cellNeighboursOfCell(mylibTag, Mesh const&, Face const& n) { return n.faces(); }

template <typename Objs, typename Init, typename Op>
auto reduce(mylibTag, Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

} // namespace mylibInterface
