//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "mylib.hpp"

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

std::ostream& toVtk(Grid const& grid, std::ostream& os) {
  os << "# vtk DataFile Version 3.0\n2D scalar data\nASCII\nDATASET "
        "UNSTRUCTURED_GRID\n";
  os << "POINTS " << grid.vertices().size() << " float\n";
  for(auto v : grid.vertices())
    os << v.x() << " " << v.y() << " 0\n";

  int fcnt = 0;
  for(auto& f : grid.faces())
    if(inner_face(f))
      ++fcnt;

  os << "CELLS " << fcnt << " " << fcnt * 4 << "\n";
  for(auto& f : grid.faces())
    if(inner_face(f))
      os << "3 " << f.vertex(0).id() << " " << f.vertex(1).id() << " " << f.vertex(2).id() << '\n';

  os << "CELL_TYPES " << fcnt << '\n';
  for(auto f : grid.faces())
    if(inner_face(f))
      os << "5\n";
  os << "CELL_DATA " << fcnt << '\n';
  return os;

  return os;
} // namespace lib_lukas
std::ostream& toVtk(std::string const& name, FaceData<double> const& f_data, Grid const& grid,
                    std::ostream& os) {
  os << "SCALARS " << name << "  float 1\nLOOKUP_TABLE default\n";
  for(auto& f : grid.faces())
    if(inner_face(f))
      os << f_data(f) << '\n';
  os << "SCALARS id int 1\nLOOKUP_TABLE default\n";
  for(auto& f : grid.faces())
    if(inner_face(f))
      os << f.id() << '\n';
  return os;
}
std::ostream& toVtk(std::string const& name, EdgeData<double> const& e_data, Grid const& grid,
                    std::ostream& os) {
  FaceData<double> f_data{grid};
  faces::reduce_on_edges(e_data, grid, wstd::identity{}, std::plus<double>{}, f_data);
  return toVtk(name, f_data, grid, os);
}
std::ostream& toVtk(std::string const& name, VertexData<double> const& v_data, Grid const& grid,
                    std::ostream& os) {
  FaceData<double> f_data{grid};
  faces::reduce_on_vertices(v_data, grid, wstd::identity{}, std::plus<double>{}, f_data);
  return toVtk(name, f_data, grid, os);
}
void Vertex::add_edge(Edge& e) { edges_.push_back(&e); }

} // namespace mylib
