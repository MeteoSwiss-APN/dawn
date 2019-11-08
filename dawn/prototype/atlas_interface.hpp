#pragma once
#include "atlas/mesh.h"
#include <cassert>

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
  T const& operator()(int f, int k) const { return atlas_field_(f, k); }
  T& operator()(int f, int k) { return atlas_field_(f, k); }

  Field(atlas::array::ArrayView<T, 2> const& atlas_field) : atlas_field_(atlas_field) {}

private:
  atlas::array::ArrayView<T, 2> atlas_field_;
};

template <typename T>
Field<T> cellFieldType(atlasTag);
template <typename T>
Field<T> edgeFieldType(atlasTag);
template <typename T>
Field<T> vertexFieldType(atlasTag);

atlas::Mesh meshType(atlasTag);

auto getCells(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.cells().size()); }
auto getEdges(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.edges().size()); }
auto getVertices(atlasTag, atlas::Mesh const& m) { return utility::irange(0, m.nodes().size()); }

// TODO Should be moved to a place common to all interfaces
template <typename Tag, typename LocationType>
auto const& deref(Tag, LocationType const& l) {
  return l;
}

std::vector<int> getNeighs(const atlas::Mesh::HybridElements::Connectivity& conn, int idx) {
  std::vector<int> neighs;
  for(int n = 0; n < conn.cols(idx); ++n) {
    neighs.emplace_back(conn(idx, n));
  }
  return neighs;
}

std::vector<int> const& cellNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    const auto& conn = m.cells().edge_connectivity();
    neighs[idx] = std::vector<int>{};
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

std::vector<int> const& edgeNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().edge_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> const& nodeNeighboursOfCell(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().node_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> cellNeighboursOfEdge(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().cell_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> nodeNeighboursOfEdge(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().node_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> cellNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().cell_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> edgeNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    neighs[idx] = getNeighs(m.cells().edge_connectivity(), idx);
  }
  return neighs[idx];
}

std::vector<int> nodeNeighboursOfNode(atlas::Mesh const& m, int const& idx) {
  // note this is only a workaround and does only work as long as we have only one mesh
  static std::map<int, std::vector<int>> neighs;
  if(neighs.count(idx) == 0) {
    const auto& conn_nodes_to_edge = m.nodes().edge_connectivity();
    neighs[idx] = std::vector<int>{};
    for(int ne = 0; ne < conn_nodes_to_edge.cols(idx); ++ne) {
      int nbh_edge_idx = conn_nodes_to_edge(idx, ne);
      const auto& conn_edge_to_nodes = m.edges().node_connectivity();
      for(int nn = 0; nn < conn_edge_to_nodes.cols(nbh_edge_idx); ++nn) {
        int nbhNode = conn_edge_to_nodes(idx, nn);
        if(nbhNode != idx) {
          neighs[idx].emplace_back();
        }
      }
    }
  }
  return neighs[idx];
}

template <typename Init, typename Op>
auto reduceCellToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : cellNeighboursOfCell(m, idx))
    op(init, obj);
  return init;
}
template <typename Init, typename Op>
auto reduceEdgeToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : edgeNeighboursOfCell(m, idx))
    op(init, obj);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToCell(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : nodeNeighboursOfCell(m, idx))
    op(init, obj);
  return init;
}

template <typename Init, typename Op>
auto reduceCellToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : cellNeighboursOfEdge(m, idx))
    op(init, obj);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToEdge(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : nodeNeighboursOfEdge(m, idx))
    op(init, obj);
  return init;
}

template <typename Init, typename Op>
auto reduceCellToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : cellNeighboursOfNode(m, idx))
    op(init, obj);
  return init;
}
template <typename Init, typename Op>
auto reduceEdgeToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : edgeNeighboursOfNode(m, idx))
    op(init, obj);
  return init;
}
template <typename Init, typename Op>
auto reduceVertexToVertex(atlasTag, atlas::Mesh const& m, int idx, Init init, Op&& op) {
  for(auto&& obj : nodeNeighboursOfNode(m, idx))
    op(init, obj);
  return init;
}

} // namespace atlasInterface
