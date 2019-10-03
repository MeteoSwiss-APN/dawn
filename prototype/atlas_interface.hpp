#pragma once

#include "atlas/mesh.h"

namespace utility {
namespace impl_ {
template <typename Integer>
class irange_ {
public:
  class iterator {
  public:
    Integer operator*() const { return i_; }
    const iterator& operator++() {
      ++i_;
      return *this;
    }
    iterator operator++(int) {
      iterator copy(*this);
      ++i_;
      return copy;
    }

    bool operator==(const iterator& other) const { return i_ == other.i_; }
    bool operator!=(const iterator& other) const { return i_ != other.i_; }

    iterator(Integer start) : i_(start) {}

  private:
    Integer i_;
  };

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }
  irange_(Integer begin, Integer end) : begin_(begin), end_(end) {}

private:
  iterator begin_;
  iterator end_;
};
} // namespace impl_
template <typename Integer>
impl_::irange_<Integer> irange(Integer from, Integer to) {
  return {from, to};
}
} // namespace utility

namespace atlasInterface {

struct atlasTag {};

template <typename T>
class Field {
public:
  T operator()(int f) const { return atlas_field_(f, 0); }
  T& operator()(int f) { return atlas_field_(f, 0); }

  Field(atlas::array::ArrayView<T, 2> const& atlas_field) : atlas_field_(atlas_field) {}

private:
  atlas::array::ArrayView<T, 2> atlas_field_;
};

auto getCells(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.cells().size()); }

class int_holder {
public:
  int& operator*() { return i_; }
  int operator*() const { return i_; }
  int_holder(int i) : i_(i) {}

private:
  int i_;
};

std::vector<int_holder> const& cellNeighboursOfCell(atlasTag, atlas::Mesh const& m,
                                                    int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int_holder>> neighs;
  if(neighs.count(idx) == 0) {
    const auto& conn = m.cells().edge_connectivity();
    neighs[idx] = std::vector<int_holder>{};
    for(int n = 0; n < conn.cols(idx); ++n) {
      int initialEdge = conn(idx, n);
      for(int c1 = 0; c1 < m.cells().size(); ++c1) {
        for(int n1 = 0; n1 < conn.cols(c1); ++n1) {
          int compareEdge = conn(c1, n1);
          if(initialEdge == compareEdge && c1 != idx) {
            neighs[idx].emplace_back(c1);
          }
        }
      }
    }
  }
  return neighs[idx];
}

template <typename Objs, typename Init, typename Op>
auto reduce(atlasTag, Objs&& objs, Init init, Op&& op) {
  for(auto&& obj : objs)
    op(init, *obj);
  return init;
}

} // namespace atlasInterface
