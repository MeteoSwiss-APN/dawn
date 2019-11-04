#include "mylib.hpp"

#include "mylib_interface.hpp"

namespace mylib {

Edge const& Vertex::edge(size_t i) const { return *edges_[i]; }
Face const& Vertex::face(size_t i) const { return *faces_[i]; }
Vertex const& Vertex::vertex(size_t i) const {
  return edge(i).vertex(0).id() == id() ? edge(i).vertex(1) : edge(i).vertex(0);
}
std::vector<const Vertex*> Vertex::vertices() const {
  std::vector<const Vertex*> ret;
  for(auto& e : edges_)
    ret.push_back(e->vertex(0).id() == id() ? &e->vertex(1) : &e->vertex(0));
  return ret;
}

Face const& Face::face(size_t i) const {
  return edge(i).face(0).id() == id() ? edge(i).face(1) : edge(i).face(0);
}
Vertex const& Face::vertex(size_t i) const { return *vertices_[i]; }
Edge const& Face::edge(size_t i) const { return *edges_[i]; }
std::vector<const Face*> Face::faces() const {
  std::vector<const Face*> ret;
  for(auto& e : edges_)
    if(&e->face(0) && &e->face(1))
      ret.push_back(e->face(0).id() == id() ? &e->face(1) : &e->face(0));
  return ret;
}

Vertex const& Edge::vertex(size_t i) const { return *vertices_[i]; }
Face const& Edge::face(size_t i) const { return *faces_[i]; }

int count_inner_faces(Grid const& grid) {
  int fcnt = 0;
  for(auto& f : grid.faces()) {
    if(inner_face(f)) {
      ++fcnt;
    }
  }
  return fcnt;
}

std::ostream& toVtk(Grid const& grid, int k_size, std::ostream& os) {
  os << "# vtk DataFile Version 3.0\n2D scalar data\nASCII\nDATASET "
        "UNSTRUCTURED_GRID\n";

  os << "POINTS " << grid.vertices().size() * k_size << " float\n";
  for(int k_level = 0; k_level < k_size; k_level++) {
    for(auto v : grid.vertices())
      os << v.x() << " " << v.y() << " " << k_level << "\n";
  }

  int fcnt = count_inner_faces(grid);

  os << "CELLS " << fcnt * k_size << " " << 4 * fcnt * k_size << "\n";
  for(int k_level = 0; k_level < k_size; k_level++) {
    for(auto& f : grid.faces())
      if(inner_face(f)) {
        const int k_offset = k_level * grid.nx() * grid.ny();
        os << "3 " << f.vertex(0).id() + k_offset << " " << f.vertex(1).id() + k_offset << " "
           << f.vertex(2).id() + k_offset << '\n';
      }
  }

  os << "CELL_TYPES " << fcnt * k_size << '\n';
  for(int k_level = 0; k_level < k_size; k_level++) {
    for(auto f : grid.faces()) {
      if(inner_face(f)) {
        os << "5\n";
      }
    }
  }

  os << "CELL_DATA " << fcnt * k_size << '\n';
  return os;
} // namespace lib_lukas

std::ostream& toVtk(std::string const& name, FaceData<double> const& f_data, Grid const& grid,
                    std::ostream& os) {
  os << "SCALARS " << name << "  float 1\nLOOKUP_TABLE default\n";
  for(int k_level = 0; k_level < f_data.k_size(); k_level++) {
    for(auto& f : grid.faces())
      if(inner_face(f))
        os << f_data(f, k_level) << '\n';
  }

  int fcnt = count_inner_faces(grid);

  os << "SCALARS id int 1\nLOOKUP_TABLE default\n";
  for(int k_level = 0; k_level < f_data.k_size(); k_level++) {
    for(auto& f : grid.faces()) {
      if(inner_face(f)) {
        os << f.id() << '\n';
      }
    }
  }
  return os;
}
std::ostream& toVtk(std::string const& name, EdgeData<double> const& e_data, Grid const& grid,
                    std::ostream& os) {
  FaceData<double> f_data{grid, e_data.k_size()};

  for(int k_level = 0; k_level < f_data.k_size(); ++k_level) {
    for(auto& cell : grid.faces()) {
      f_data(cell, k_level) = mylibInterface::reduceEdgeToCell(
          mylibInterface::mylibTag{}, grid, cell, 0,
          [&](auto& lhs, const auto& rhs) { lhs += f_data(cell, k_level); });
    }
  }

  return toVtk(name, f_data, grid, os);
}
std::ostream& toVtk(std::string const& name, VertexData<double> const& v_data, Grid const& grid,
                    std::ostream& os) {
  FaceData<double> f_data{grid, v_data.k_size()};

  for(int k_level = 0; k_level < f_data.k_size(); ++k_level) {
    for(auto& cell : grid.faces()) {
      f_data(cell, k_level) = mylibInterface::reduceVertexToCell(
          mylibInterface::mylibTag{}, grid, cell, 0,
          [&](auto& lhs, const auto& rhs) { lhs += f_data(cell, k_level); });
    }
  }

  return toVtk(name, f_data, grid, os);
}
void Vertex::add_edge(Edge& e) { edges_.push_back(&e); }

} // namespace mylib
