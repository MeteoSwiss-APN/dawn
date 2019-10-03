#pragma once

#include "mylib.hpp"

namespace mylibInterface {

struct mylibTag {};

decltype(auto) getCells(mylibTag, mylib::Grid const& m) { return m.faces(); }

decltype(auto) cellNeighboursOfCell(mylibTag, mylib::Grid const&, mylib::Face const& n) {
  return n.faces();
}

template <typename Objs, typename Init, typename Op>
auto reduce(mylibTag, Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

} // namespace mylibInterface
