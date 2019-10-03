#pragma once

#include <cmath>
#include <iostream>
#include <vector>

namespace wstd {
struct identity {
  template <class T>
  constexpr T&& operator()(T&& t) const noexcept {
    return std::forward<T>(t);
  }
};
} // namespace wstd

namespace lib_lukas {
class Vertex;
class Edge;
class Face;

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
        if(j > 0 || periodic) //
          v.add_face(face_at(i, j - 1, face_color::upward));
        if(i < nx && j < ny || periodic) {
          v.add_face(face_at(i, j, face_color::downward));
          v.add_face(face_at(i, j, face_color::upward));
        }
        v.add_face(face_at(i - 1, j, face_color::downward));
      }
  }

  std::vector<Face> const& faces() const { return faces_; }
  std::vector<Vertex> const& vertices() const { return vertices_; }
  std::vector<Edge> const& edges() const { return edges_; }

  auto nx() const { return nx_; }
  auto ny() const { return ny_; }

private:
  std::vector<Face> faces_;
  std::vector<Vertex> vertices_;
  std::vector<Edge> edges_;

  int nx_;
  int ny_;
};

template <typename O, typename T>
class Data {
public:
  explicit Data(size_t size) : data_(size) {}
  T& operator[](O const& f) { return data_[f.id()]; }
  T const& operator[](O const& f) const { return data_[f.id()]; }
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }

private:
  std::vector<T> data_;
};
template <typename T>
class FaceData : public Data<Face, T> {
public:
  explicit FaceData(Grid const& grid) : Data<Face, T>(grid.faces().size()) {}
};
template <typename T>
class VertexData : public Data<Vertex, T> {
public:
  explicit VertexData(Grid const& grid) : Data<Vertex, T>(grid.vertices().size()) {}
};
template <typename T>
class EdgeData : public Data<Edge, T> {
public:
  explicit EdgeData(Grid const& grid) : Data<Edge, T>(grid.edges().size()) {}
};
template <typename T>
inline T gauss(T width, T x, T y, T nx, T ny, bool f = false) {
  return std::exp(-width * (std::pow(x - nx / 2.0, 2) + std::pow(y - ny / 2.0, 2)));
}
template <typename T>
inline void init_gaussian(T width, VertexData<T>& v_data, Grid const& grid) {
  for(auto& v : grid.vertices())
    v_data[v] = gauss(width, v.x(), v.y(), (T)grid.nx(), (T)grid.ny());
}
template <typename T>
inline void init_gaussian(T width, EdgeData<T>& e_data, Grid const& grid) {
  for(auto& e : grid.edges())
    if(e) {
      auto center_x = 0.5 * (e.vertex(0).x() + e.vertex(1).x());
      auto center_y = 0.5 * (e.vertex(0).y() + e.vertex(1).y());

      e_data[e] = gauss(width, center_x, center_y, (T)grid.nx(), (T)grid.ny());
    }
}
template <typename T>
inline void init_gaussian(T width, FaceData<T>& f_data, Grid const& grid) {
  for(auto& f : grid.faces()) {
    auto center_x = (1.f / 3) * (f.vertex(0).x() + f.vertex(1).x() + f.vertex(2).x());
    auto center_y = (1.f / 3) * (f.vertex(0).y() + f.vertex(1).y() + f.vertex(2).y());

    f_data[f] = gauss(width, center_x, center_y, (T)grid.nx(), (T)grid.ny(), true);
  }
}
namespace faces {
template <class T, class Map, class Reduce>
auto reduce_on_vertices(VertexData<T> const& v_data, Grid const& grid, Map const& map,
                        Reduce const& reduce, FaceData<T> ret) {
  for(auto f : grid.faces()) {
    for(auto v : f.vertices())
      ret[f] = reduce(ret[f], map(v_data[*v]));
    ret[f] /= f.vertices().size();
  }
  return ret;
}
inline int edge_sign(Edge const& curr, Edge const& prev) {
  auto p0 = prev.vertex(0).id();
  auto p1 = prev.vertex(1).id();

  auto c0 = curr.vertex(0).id();
  auto c1 = curr.vertex(1).id();

  auto start = c0 == p0 || c0 == p1 ? c0 : c1;
  auto end = c0 == p0 || c0 == p1 ? c1 : c0;
  return start < end ? 1 : -1;
}
template <class T, class Map, class Reduce>
std::enable_if_t<(sizeof(std::declval<Map>()(std::declval<T>(), 1)) > 0), FaceData<float>>
reduce_on_edges(EdgeData<T> const& e_data, Grid const& grid, Map const& map, Reduce const& reduce,
                FaceData<float> ret) {
  for(auto f : grid.faces()) {
    // inward, if edge from negative to positive
    //   .
    //   |\
        //  0| \ 1   0: outwards
    //   |  \    1: inwards
    //   ----'   2: outwards
    //     2
    //
    //     1
    //   ----
    //   \  |
    //  0 \ |2   0: outwards
    //     \|    1: inwards
    //      ^    2: inwards
    auto edges = f.edges();
    for(auto prev = edges.end() - 1, curr = edges.begin(); curr != edges.end(); prev = curr, ++curr)
      ret[f] = reduce(ret[f], map(e_data[**curr], edge_sign(**curr, **prev)));
    ret[f] /= f.edges().size();
  }
  return ret;
}
template <class T, class Map, class Reduce>
std::enable_if_t<(sizeof(std::declval<Map>()(std::declval<T>())) > 0), FaceData<T>>
reduce_on_edges(EdgeData<T> const& e_data, Grid const& grid, Map const& map, Reduce const& reduce,
                FaceData<T> ret) {
  for(auto f : grid.faces()) {
    for(auto e : f.edges())
      ret[f] = reduce(ret[f], map(e_data[*e]));
    ret[f] /= f.edges().size();
  }
  return ret;
}

template <class T, class Map, class Reduce>
auto reduce_on_faces(FaceData<T> const& f_data, Grid const& grid, Map const& map,
                     Reduce const& reduce, FaceData<T> ret) {
  for(auto f : grid.faces()) {
    for(auto next_f : f.faces())
      ret[f] = reduce(ret[f], map(f_data[*next_f]));
    ret[f] /= f.faces().size();
  }
  return ret;
}

} // namespace faces

namespace edges {
template <class T, class Map, class Reduce>
auto reduce_on_faces(FaceData<T> const& f_data, Grid const& grid, Map const& map,
                     Reduce const& reduce, EdgeData<T> ret) {
  for(auto e : grid.edges()) {
    if(e.faces().size() == 2) {
      ret[e] = reduce(ret[e], map(f_data[e.face(0)], 1));
      ret[e] = reduce(ret[e], map(f_data[e.face(1)], -1));
    } else
      ret[e] = 0;
  }
  return ret;
}
} // namespace edges
std::ostream& toVtk(Grid const& grid, std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, FaceData<double> const& f_data, Grid const& grid,
                    std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, EdgeData<double> const& e_data, Grid const& grid,
                    std::ostream& os = std::cout);
std::ostream& toVtk(std::string const& name, VertexData<double> const& v_data, Grid const& grid,
                    std::ostream& os = std::cout);
} // namespace lib_lukas
using namespace lib_lukas;
