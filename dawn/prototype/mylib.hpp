#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

namespace mylib {
class Vertex;
class Edge;
class Face;

//   .---.---.---.
//   |\ 1|\ 3|\ 5|
//   | \ | \ | \ |
//   |0 \|2 \|4 \|
//   ----'---'---'
enum face_color { upward = 0, downward = 1 };
enum edge_color { horizontal = 0, diagonal = 1, vertical = 2 };

class Vertex {
public:
  Vertex() = default;
  Vertex(int x, int y, int id) : x_(x), y_(y), id_(id) {}

  int x() const { return x_; }
  int y() const { return y_; }
  int id() const { return id_; }

  Edge const& edge(size_t i) const;
  Face const& face(size_t i) const;
  Vertex const& vertex(size_t i) const;
  auto edges() const { return edges_; }
  auto faces() const { return faces_; }
  std::vector<const Vertex*> vertices() const;

  void add_edge(Edge& e);
  void add_face(Face& f) { faces_.push_back(&f); }

private:
  int id_;
  int x_;
  int y_;

  std::vector<Edge*> edges_;
  std::vector<Face*> faces_;
};
class Face {
public:
  Face() = default;
  Face(int id, face_color color) : id_(id), color_(color) {}

  int id() const { return id_; }
  face_color color() const { return color_; }

  Vertex const& vertex(size_t i) const;
  Edge const& edge(size_t i) const;
  Face const& face(size_t i) const;
  auto vertices() const { return vertices_; }
  auto edges() const { return edges_; }
  std::vector<const Face*> faces() const;

  void add_edge(Edge& e) { edges_.push_back(&e); }
  void add_vertex(Vertex& v) { vertices_.push_back(&v); }

private:
  int id_;
  face_color color_;

  std::vector<Edge*> edges_;
  std::vector<Vertex*> vertices_;
};
class Edge {
public:
  Edge() = default;
  Edge(int id, edge_color color) : id_(id), color_(color) {}

  int id() const { return id_; }
  edge_color color() const { return color_; }

  Vertex const& vertex(size_t i) const;
  Face const& face(size_t i) const;
  auto faces() const { return faces_; }
  auto vertices() const { return vertices_; }

  void add_vertex(Vertex& v) { vertices_.push_back(&v); }
  void add_face(Face& f) { faces_.push_back(&f); }

  operator bool() const { return id_ >= 0; }

private:
  int id_ = -1;
  edge_color color_;

  std::vector<Vertex*> vertices_;
  std::vector<Face*> faces_;
};

class Grid {
public:
  Grid(int nx, int ny, bool periodic = false)
      : faces_(2 * nx * ny), vertices_(periodic ? nx * ny : (nx + 1) * (ny + 1)),
        edges_(periodic ? 3 * nx * ny : 3 * (nx + 1) * (ny + 1)), nx_(nx), ny_(ny) {
    auto edge_at = [&](int i, int j, int c) -> Edge& {
      if(periodic)
        return edges_.at(3 * (((j + ny) % ny) * nx + ((i + nx) % nx)) + c);
      else
        return edges_.at(3 * (j * (nx + 1) + i) + c);
    };
    auto vertex_at = [&](int i, int j) -> Vertex& {
      if(periodic)
        return vertices_.at(((j + ny) % ny) * nx + ((i + nx) % nx));
      else
        return vertices_.at(j * (nx + 1) + i);
    };
    auto face_at = [&](int i, int j, int c) -> Face& {
      if(periodic)
        return faces_.at(2 * (((j + ny) % ny) * nx + ((i + nx) % nx)) + c);
      else
        return faces_.at(2 * (j * nx + i) + c);
    };

    for(int j = 0; j < ny; ++j)
      for(int i = 0; i < nx; ++i)
        for(int c = 0; c < 2; ++c) {
          auto& f = face_at(i, j, c);
          f = Face(&f - faces_.data(), (face_color)c);
        }
    for(int j = 0; j < (periodic ? ny : ny + 1); ++j)
      for(int i = 0; i < (periodic ? nx : nx + 1); ++i) {
        auto& v = vertex_at(i, j);
        v = Vertex(i, j, &v - vertices_.data());
      }

    for(int j = 0; j < ny; ++j)
      for(int i = 0; i < nx; ++i) {
        auto& f_uw = face_at(i, j, face_color::upward);
        //   .
        //   |\
        //  0| \ 1
        //   |  \
        //   ----'
        //     2
        f_uw.add_edge(edge_at(i, j, edge_color::vertical));
        f_uw.add_edge(edge_at(i, j, edge_color::diagonal));
        f_uw.add_edge(edge_at(i, j + 1, edge_color::horizontal));

        //   0
        //   .
        //   |\
        //   | \
        //   |  \
        //   ----' 1
        //  2
        f_uw.add_vertex(vertex_at(i, j));
        f_uw.add_vertex(vertex_at(i + 1, j + 1));
        f_uw.add_vertex(vertex_at(i, j + 1));

        // downward
        auto& f_dw = face_at(i, j, face_color::downward);
        //     1
        //   ----
        //   \  |
        //  0 \ |2
        //     \|
        //      ^
        f_dw.add_edge(edge_at(i, j, edge_color::diagonal));
        f_dw.add_edge(edge_at(i, j, edge_color::horizontal));
        f_dw.add_edge(edge_at(i + 1, j, edge_color::vertical));

        //        1
        // 0 ----
        //   \  |
        //    \ |
        //     \|
        //      ^ 2
        f_dw.add_vertex(vertex_at(i, j));
        f_dw.add_vertex(vertex_at(i + 1, j));
        f_dw.add_vertex(vertex_at(i + 1, j + 1));
      }

    for(int j = 0; j < (periodic ? ny : ny + 1); ++j)
      for(int i = 0; i < nx; ++i) {
        //     0
        // 0 ----- 1
        //     1
        auto& e = edge_at(i, j, edge_color::horizontal);
        e = Edge(&e - edges_.data(), edge_color::horizontal);
        e.add_vertex(vertex_at(i, j));
        e.add_vertex(vertex_at(i + 1, j));

        if(j > 0 || periodic)
          e.add_face(face_at(i, j - 1, face_color::upward));
        if(j < ny || periodic)
          e.add_face(face_at(i, j, face_color::downward));
      }
    for(int j = 0; j < ny; ++j)
      for(int i = 0; i < nx; ++i) {
        // 0
        //  \  0
        //   \
        // 1  \
        //     1
        auto& e = edge_at(i, j, edge_color::diagonal);
        e = Edge(&e - edges_.data(), edge_color::diagonal);
        e.add_vertex(vertex_at(i, j));
        e.add_vertex(vertex_at(i + 1, j + 1));

        e.add_face(face_at(i, j, face_color::downward));
        e.add_face(face_at(i, j, face_color::upward));
      }
    for(int j = 0; j < ny; ++j)
      for(int i = 0; i < (periodic ? nx : nx + 1); ++i) {
        //     0
        // \^^^|\
        //  \1 | \
        //   \ | 0\
        //    \|___\
        //     1
        auto& e = edge_at(i, j, edge_color::vertical);
        e = Edge(&e - edges_.data(), edge_color::vertical);
        e.add_vertex(vertex_at(i, j));
        e.add_vertex(vertex_at(i, j + 1));

        if(i < nx || periodic)
          e.add_face(face_at(i, j, face_color::upward));
        if(i > 0 || periodic)
          e.add_face(face_at(i - 1, j, face_color::downward));
      }

    for(int j = 0; j < (periodic ? ny : ny + 1); ++j)
      for(int i = 0; i < (periodic ? nx : nx + 1); ++i) {
        auto& v = vertex_at(i, j);
        //  1   2
        //   \  |
        //    \ |
        //     \|
        // 0---------- 3
        //      |\
        //      | \
        //      |  \
        //      5   4
        if(i > 0 || periodic) //
          v.add_edge(edge_at(i - 1, j, edge_color::horizontal));
        if(i > 0 && j > 0 || periodic) //
          v.add_edge(edge_at(i - 1, j - 1, edge_color::diagonal));
        if(j > 0 || periodic) //
          v.add_edge(edge_at(i, j - 1, edge_color::vertical));
        if(i < nx || periodic) //
          v.add_edge(edge_at(i, j, edge_color::horizontal));
        if(i < nx && j < ny || periodic) //
          v.add_edge(edge_at(i, j, edge_color::diagonal));
        if(j < ny || periodic) //
          v.add_edge(edge_at(i, j, edge_color::vertical));

        //    1
        //   \  |
        // 0  \ |  2
        //     \|
        //  ----------
        //      |\
        //   5  | \ 3
        //      |  \
        //        4
        if(i > 0 && j > 0 || periodic) {
          v.add_face(face_at(i - 1, j - 1, face_color::upward));
          v.add_face(face_at(i - 1, j - 1, face_color::downward));
        }
        if(i < nx && j > 0 || periodic) //
          v.add_face(face_at(i, j - 1, face_color::upward));
        if(i < nx && j < ny || periodic) {
          v.add_face(face_at(i, j, face_color::downward));
          v.add_face(face_at(i, j, face_color::upward));
        }
        if(i > 0 && j < ny || periodic) {
          v.add_face(face_at(i - 1, j, face_color::downward));
        }
      }

    for(auto const& e : edges_) {
      if(e.id() != -1)
        valid_edges_.push_back(e);
    }
  }

  std::vector<Face> const& faces() const { return faces_; }
  std::vector<Vertex> const& vertices() const { return vertices_; }
  // edges_ contains edges outside of the domain, these are removed in valid_edges_.
  std::vector<std::reference_wrapper<Edge const>> const& edges() const { return valid_edges_; }
  std::vector<Edge> const& all_edges() const { return edges_; }

  auto nx() const { return nx_; }
  auto ny() const { return ny_; }

private:
  std::vector<Face> faces_;
  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;
  std::vector<std::reference_wrapper<Edge const>> valid_edges_;

  int nx_;
  int ny_;
};

//===------------------------------------------------------------------------------------------===//
// dense fields
//===------------------------------------------------------------------------------------------===//

template <typename O, typename T>
class Data {
public:
  Data(size_t horizontal_size, size_t num_k_levels)
      : data_(num_k_levels, std::vector<T>(horizontal_size)) {}
  T& operator()(O const& f, size_t k_level) { return data_[k_level][f.id()]; }
  T const& operator()(O const& f, size_t k_level) const { return data_[k_level][f.id()]; }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }

  int k_size() const { return data_.size(); }

private:
  std::vector<std::vector<T>> data_;
};

template <typename T>
class FaceData : public Data<Face, T> {
public:
  FaceData(Grid const& grid, int k_size) : Data<Face, T>(grid.faces().size(), k_size) {}
};
template <typename T>
class VertexData : public Data<Vertex, T> {
public:
  VertexData(Grid const& grid, int k_size) : Data<Vertex, T>(grid.vertices().size(), k_size) {}
};
template <typename T>
class EdgeData : public Data<Edge, T> {
public:
  EdgeData(Grid const& grid, int k_size) : Data<Edge, T>(grid.all_edges().size(), k_size) {}
};

//===------------------------------------------------------------------------------------------===//
// sparse fields
//===------------------------------------------------------------------------------------------===//

template <typename O, typename T>
class SparseData {
public:
  SparseData(size_t num_k_levels, size_t dense_size, size_t sparse_size)
      : data_(num_k_levels, std::vector<std::vector<T>>(dense_size, std::vector<T>(sparse_size))) {}
  T& operator()(const O& elem, size_t sparse_idx, size_t k_level) {
    return data_[k_level][elem.id()][sparse_idx];
  }
  T const& operator()(const O& elem, size_t sparse_idx, size_t k_level) const {
    return data_[k_level][elem.id()][sparse_idx];
  }
  int k_size() const { return data_.size(); }

private:
  std::vector<std::vector<std::vector<T>>> data_;
};

template <typename T>
class SparseFaceData : public SparseData<Face, T> {
public:
  SparseFaceData(Grid const& grid, int k_size, int sparse_size)
      : SparseData<Face, T>(k_size, grid.faces().size(), sparse_size) {}
};
template <typename T>
class SparseVertexData : public SparseData<Vertex, T> {
public:
  SparseVertexData(Grid const& grid, int k_size, int sparse_size)
      : SparseData<Vertex, T>(k_size, grid.vertices().size(), sparse_size) {}
};
template <typename T>
class SparseEdgeData : public SparseData<Edge, T> {
public:
  SparseEdgeData(Grid const& grid, int k_size, int sparse_size)
      : SparseData<Edge, T>(k_size, grid.all_edges().size(), sparse_size) {}
};

std::ostream& toVtk(Grid const& grid, int k_size, std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, FaceData<double> const& f_data, Grid const& grid,
                    std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, EdgeData<double> const& e_data, Grid const& grid,
                    std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, VertexData<double> const& v_data, Grid const& grid,
                    std::ostream& os = std::cout);

namespace {
bool inner_face(Face const& f) {
  return (f.color() == face_color::downward && f.vertex(0).id() < f.vertex(1).id() &&
          f.vertex(0).id() < f.vertex(2).id()) ||
         (f.color() == face_color::upward && f.vertex(1).id() > f.vertex(0).id() &&
          f.vertex(1).id() > f.vertex(2).id());
}
} // namespace

} // namespace mylib
