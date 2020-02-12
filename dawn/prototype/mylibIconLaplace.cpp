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
#include "mylib_interface.hpp"
#include <assert.h>
#include <fstream>
#include <optional>

// re-implemenation of the ICON nabla2_vec operator.
//    see mo_math_divrot.f90 and mo_math_laplace.f90
// names have been kept close to the FORTRAN code, but the "_Location" suffixes have been removed
// because of the strong typing in C++ and inconsistent application in the FORTRAN source

//===------------------------------------------------------------------------------------------===//
// geometric helper functions
//===------------------------------------------------------------------------------------------===//

template <typename T>
static int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

double EdgeLength(const mylib::Edge& e) {
  double x0 = e.vertex(0).x();
  double y0 = e.vertex(0).y();
  double x1 = e.vertex(1).x();
  double y1 = e.vertex(1).y();
  double dx = x1 - x0;
  double dy = y1 - y0;
  return sqrt(dx * dx + dy * dy);
}

std::tuple<double, double> EdgeMidpoint(const mylib::Edge& e) {
  double x0 = e.vertex(0).x();
  double y0 = e.vertex(0).y();
  double x1 = e.vertex(1).x();
  double y1 = e.vertex(1).y();
  return {0.5 * (x0 + x1), 0.5 * (y0 + y1)};
}

std::tuple<double, double> PrimalNormal(const mylib::Edge& e) {
  double l = EdgeLength(e);
  double x0 = e.vertex(0).x();
  double y0 = e.vertex(0).y();
  double x1 = e.vertex(1).x();
  double y1 = e.vertex(1).y();
  return {-(y1 - y0) / l, (x1 - x0) / l};
}

std::tuple<double, double> CellMidPoint(const mylib::Face& c) {
  auto v0 = c.vertex(0);
  auto v1 = c.vertex(1);
  auto v2 = c.vertex(2);
  return {1. / 3. * (v0.x() + v1.x() + v2.x()), 1. / 3. * (v0.y() + v1.y() + v2.y())};
}

std::tuple<double, double> CellCircumcenter(const mylib::Face& c) {
  double Ax = c.vertex(0).x();
  double Ay = c.vertex(0).y();
  double Bx = c.vertex(1).x();
  double By = c.vertex(1).y();
  double Cx = c.vertex(2).x();
  double Cy = c.vertex(2).y();

  double D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By));
  double Ux = 1. / D *
              ((Ax * Ax + Ay * Ay) * (By - Cy) + (Bx * Bx + By * By) * (Cy - Ay) +
               (Cx * Cx + Cy * Cy) * (Ay - By));
  double Uy = 1. / D *
              ((Ax * Ax + Ay * Ay) * (Cx - Bx) + (Bx * Bx + By * By) * (Ax - Cx) +
               (Cx * Cx + Cy * Cy) * (Bx - Ax));
  return {Ux, Uy};
}

double TriangleArea(const mylib::Vertex& v0, const mylib::Vertex& v1, const mylib::Vertex& v2) {
  return fabs(
      (v0.x() * (v1.y() - v2.y()) + v1.x() * (v2.y() - v0.y()) + v2.x() * (v0.y() - v1.y())) * 0.5);
}

double CellArea(const mylib::Face& c) {
  auto v0 = c.vertex(0);
  auto v1 = c.vertex(1);
  auto v2 = c.vertex(2);
  return TriangleArea(v0, v1, v2);
}

double DualCellArea(const mylib::Vertex& center) {
  double totalArea = 0.;
  for(const auto& e : center.edges()) {
    if(e->faces().size() != 2) {
      return 0.;
    }
    auto [leftx, lefty] = CellCircumcenter(e->face(0));
    auto [rightx, righty] = CellCircumcenter(e->face(1));
    mylib::Vertex left(leftx, lefty, -1);
    mylib::Vertex right(rightx, righty, -1);
    totalArea += TriangleArea(center, left, right);
  }
  return totalArea;
}

double DualEdgeLength(const mylib::Edge& e) {
  if(e.faces().size() == 1) { // dual edge length is zero on boundaries!
    return 0.;
  }
  auto c0 = e.face(0);
  auto c1 = e.face(1);
  auto [x0, y0] = CellCircumcenter(c0);
  auto [x1, y1] = CellCircumcenter(c1);
  double dx = x1 - x0;
  double dy = y1 - y0;
  return sqrt(dx * dx + dy * dy);
}

double TangentOrientation(const mylib::Edge& e) {
  // ! =1 if vector product of vector from vertex1 to vertex 2 (v2-v1) by vector
  // ! from cell c1 to cell c2 (c2-c1) goes outside the sphere
  // ! =-1 if vector product ...       goes inside  the sphere

  if(e.faces().size() == 1) { // not sure about this on the boundaries. chose 1 arbitrarily
    return 1.;
  }

  auto c0 = e.face(0);
  auto c1 = e.face(1);
  auto [x0, y0] = CellCircumcenter(c0);
  auto [x1, y1] = CellCircumcenter(c1);
  double c2c1x = x1 - x0;
  double c2c1y = y1 - y0;

  auto v0 = e.vertex(0);
  auto v1 = e.vertex(1);
  double v2v1x = v1.x() - v0.x();
  double v2v1y = v1.y() - v0.y();

  return sgn(c2c1x * v2v1y - c2c1y * v2v1x);
}

//===------------------------------------------------------------------------------------------===//
// output (debugging)
//===------------------------------------------------------------------------------------------===//
void dumpMesh(const mylib::Grid& m, const std::string& fname);
void dumpDualMesh(const mylib::Grid& m, const std::string& fname);

void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseVertexData<double>& sparseData,
                    int level, int edgesPerVertex, const std::string& fname);
void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseFaceData<double>& sparseData,
                    int level, int edgesPerCell, const std::string& fname);

void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::EdgeData<double>& field, int level,
               std::optional<mylib::edge_color> color = std::nullopt);
void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::EdgeData<double>& field_x, const mylib::EdgeData<double>& field_y,
               int level, std::optional<mylib::edge_color> color = std::nullopt);
void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::FaceData<double>& field, int level,
               std::optional<mylib::face_color> color = std::nullopt);
void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::VertexData<double>& field, int level);

int main() {
  int w = 20;
  int k_size = 1;
  const int level = 0;
  double lDomain = M_PI;
  // double lDomain = 10.;

  const bool dbg_out = true;

  mylib::Grid mesh{w, w, false, lDomain, lDomain, true};
  if(dbg_out) {
    dumpMesh(mesh, "laplICONmylib_mesh.txt");
    dumpDualMesh(mesh, "laplICONmylib_dualMesh.txt");
  }

  const int edgesPerVertex = 6;
  const int edgesPerCell = 3;

  //===------------------------------------------------------------------------------------------===//
  // input field (field we want to take the laplacian of)
  //===------------------------------------------------------------------------------------------===//
  mylib::EdgeData<double> vec(mesh, k_size);
  //    this is, confusingly, called vec_e, even though it is a scalar
  //    _conceptually_, this can be regarded as a vector with implicit direction (presumably normal
  //    to edge direction)

  //===------------------------------------------------------------------------------------------===//
  // output field (field containing the computed laplacian)
  //===------------------------------------------------------------------------------------------===//
  mylib::EdgeData<double> nabla2_vec(mesh, k_size);
  mylib::EdgeData<double> nabla2t1_vec(mesh, k_size); // term 1 and term 2 of nabla for debugging
  mylib::EdgeData<double> nabla2t2_vec(mesh, k_size);
  //    again, surprisingly enough, this is a scalar quantity even though the vector laplacian is a
  //    laplacian.

  //===------------------------------------------------------------------------------------------===//
  // intermediary fields (curl/rot and div of vec_e)
  //===------------------------------------------------------------------------------------------===//

  // rotation (more commonly curl) of vec_e on vertices
  //    I'm not entirely positive how one can take the curl of a scalar field (commonly a undefined
  //    operation), however, since vec_e is _conceptually_ a vector this works out. somehow.
  mylib::VertexData<double> rot_vec(mesh, k_size);

  // divergence of vec_e on cells
  //    Again, not entirely sure how one can measure the divergence of scalars, but again, vec_e is
  //    _conceptually_ a vector, so...
  mylib::FaceData<double> div_vec(mesh, k_size);

  //===------------------------------------------------------------------------------------------===//
  // sparse dimensions for computing intermediary fields
  //===------------------------------------------------------------------------------------------===//

  // needed for the computation of the curl/rotation. according to documentation this needs to be:
  // ! the appropriate dual cell based verts%edge_orientation
  // ! is required to obtain the correct value for the
  // ! application of Stokes theorem (which requires the scalar
  // ! product of the vector field with the tangent unit vectors
  // ! going around dual cell jv COUNTERCLOKWISE;
  // ! since the positive direction for the vec_e components is
  // ! not necessarily the one yelding counterclockwise rotation
  // ! around dual cell jv, a correction coefficient (equal to +-1)
  // ! is necessary, given by g%verts%edge_orientation
  mylib::SparseVertexData<double> geofac_rot(mesh, k_size, edgesPerVertex);
  mylib::SparseVertexData<double> edge_orientation_vertex(mesh, k_size, edgesPerVertex);

  // needed for the computation of the curl/rotation. according to documentation this needs to be:
  //   ! ...the appropriate cell based edge_orientation is required to
  //   ! obtain the correct value for the application of Gauss theorem
  //   ! (which requires the scalar product of the vector field with the
  //   ! OUTWARD pointing unit vector with respect to cell jc; since the
  //   ! positive direction for the vector components is not necessarily
  //   ! the outward pointing one with respect to cell jc, a correction
  //   ! coefficient (equal to +-1) is necessary, given by
  //   ! ptr_patch%grid%cells%edge_orientation)
  mylib::SparseFaceData<double> geofac_div(mesh, k_size, edgesPerCell);
  mylib::SparseFaceData<double> edge_orientation_cell(mesh, k_size, edgesPerCell);

  //===------------------------------------------------------------------------------------------===//
  // fields containing geometric information
  //===------------------------------------------------------------------------------------------===//
  mylib::EdgeData<double> tangent_orientation(mesh, k_size);
  mylib::EdgeData<double> primal_edge_length(mesh, k_size);
  mylib::EdgeData<double> dual_edge_length(mesh, k_size);
  mylib::EdgeData<double> dual_normal_x(mesh, k_size);
  mylib::EdgeData<double> dual_normal_y(mesh, k_size);
  mylib::EdgeData<double> primal_normal_x(mesh, k_size);
  mylib::EdgeData<double> primal_normal_y(mesh, k_size);

  mylib::FaceData<double> cell_area(mesh, k_size);

  mylib::VertexData<double> dual_cell_area(mesh, k_size);

  //===------------------------------------------------------------------------------------------===//
  // initialize fields
  //===------------------------------------------------------------------------------------------===//

  for(const auto& v : mesh.vertices()) {
    rot_vec(v, level) = 0;
  }

  for(const auto& c : mesh.faces()) {
    div_vec(c, level) = 0;
  }

  // init geometric info for edges
  for(auto const& e : mesh.edges()) {
    primal_edge_length(e, level) = EdgeLength(e);
    dual_edge_length(e, level) = DualEdgeLength(e);
    tangent_orientation(e, level) = TangentOrientation(e);
    auto [xm, ym] = EdgeMidpoint(e);
    auto [nx, ny] = PrimalNormal(e);
    primal_normal_x(e, level) = nx;
    primal_normal_y(e, level) = ny;
    // The primal normal, dual normal
    // forms a left-handed coordinate system
    dual_normal_x(e, level) = ny;
    dual_normal_y(e, level) = -nx;
  }
  if(dbg_out) {
    dumpField("laplICONmylib_EdgeLength.txt", mesh, primal_edge_length, level);
    dumpField("laplICONmylib_dualEdgeLength.txt", mesh, dual_edge_length, level);
    dumpField("laplICONmylib_nrm.txt", mesh, primal_normal_x, primal_normal_y, level);
    dumpField("laplICONmylib_dnrm.txt", mesh, dual_normal_x, dual_normal_y, level);
  }

  auto wave = [](double px, double py) { return sin(px) * sin(py); };
  auto constant = [](double px, double py) { return 1.; };
  auto lin = [](double px, double py) { return px; };

  // init zero and test function
  for(const auto& e : mesh.edges()) {
    auto [xm, ym] = EdgeMidpoint(e);
    double py = 2 / sqrt(3) * ym;
    double px = xm + 0.5 * py;

    // this way to initialize the field is wrong, or at least does it does not correspond to what
    // one might expect intuitively. the values on the edges are the lengths of vectors in the
    // direction of the edge normal. assigning a constant field would thus mean that quantity 1
    // flows into the cell on two edges, and out on another (or vice versa). Divergence will hence
    // not be zero in this case!
    double fun = wave(px, py);
    vec(e.get(), level) = fun;

    nabla2_vec(e.get(), level) = 0;
  }

  if(dbg_out) {
    dumpField("laplICONmylib_in.txt", mesh, vec, level);
  }

  // init geometric info for cells
  for(const auto& c : mesh.faces()) {
    cell_area(c, level) = CellArea(c);
  }
  // init geometric info for vertices
  for(const auto& v : mesh.vertices()) {
    dual_cell_area(v, level) = DualCellArea(v);
  }

  if(dbg_out) {
    dumpField("laplICONmylib_areaCell.txt", mesh, cell_area, level);
    dumpField("laplICONmylib_areaCellDual.txt", mesh, dual_cell_area, level);
  }

  // init edge orientations for vertices and cells
  auto dot = [](const mylib::Vertex& v1, const mylib::Vertex& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y();
  };

  // +1 when the vector from this to the neigh-
  // bor vertex has the same orientation as the
  // tangent unit vector of the connecting edge.
  // -1 otherwise
  for(const auto& v : mesh.vertices()) {
    int m_sparse = 0;
    for(const auto& e : v.edges()) {
      mylib::Vertex testVec =
          mylib::Vertex(v.vertex(m_sparse).x() - v.x(), v.vertex(m_sparse).y() - v.y(), -1);
      mylib::Vertex dual = mylib::Vertex(dual_normal_x(*e, level), dual_normal_y(*e, level), -1);
      edge_orientation_vertex(v, m_sparse, level) = sgn(dot(testVec, dual));
      m_sparse++;
    }
  }

  // The orientation of the edge normal vector
  // (the variable primal normal in the edges ta-
  // ble) for the cell according to Gauss formula.
  // It is equal to +1 if the normal to the edge
  // is outwards from the cell, otherwise is -1.
  for(const auto& c : mesh.faces()) {
    int m_sparse = 0;
    auto [xm, ym] = CellCircumcenter(c);
    for(const auto& e : c.edges()) {
      mylib::Vertex vOutside(e->vertex(0).x() - xm, e->vertex(0).y() - ym, -1);
      edge_orientation_cell(c, m_sparse, level) =
          sgn(dot(mylib::Vertex(e->vertex(0).x() - xm, e->vertex(0).y() - ym, -1),
                  mylib::Vertex(primal_normal_x(*e, level), primal_normal_y(*e, level), -1)));
      m_sparse++;
      // explanation: the vector cellMidpoint -> e.vertex(0) is guaranteed to point outside. The dot
      // product checks if the edge normal has the same orientation. e.vertex(0) is arbitrary,
      // vertex(1), or any point on e would work just as well
    }
  }

  // init sparse quantities for div and rot
  for(const auto& v : mesh.vertices()) {
    int m_sparse = 0;
    for(const auto& e : v.edges()) {
      geofac_rot(v, m_sparse, level) = dual_edge_length(*e, level) *
                                       edge_orientation_vertex(v, m_sparse, level) /
                                       dual_cell_area(v, level);

      // ptr_int%geofac_rot(jv,je,jb) =                &
      //    & ptr_patch%edges%dual_edge_length(ile,ibe) * &
      //    & ptr_patch%verts%edge_orientation(jv,jb,je)/ &
      //    & ptr_patch%verts%dual_area(jv,jb) * REAL(ifac,wp)
      m_sparse++;
    }
  }

  for(const auto& c : mesh.faces()) {
    int m_sparse = 0;
    for(const auto& e : c.edges()) {
      geofac_div(c, m_sparse, level) = primal_edge_length(*e, level) *
                                       edge_orientation_cell(c, m_sparse, level) /
                                       cell_area(c, level);

      //  ptr_int%geofac_div(jc,je,jb) = &
      //    & ptr_patch%edges%primal_edge_length(ile,ibe) * &
      //    & ptr_patch%cells%edge_orientation(jc,jb,je)  / &
      //    & ptr_patch%cells%area(jc,jb)
      m_sparse++;
    }
  }

  if(dbg_out) {
    dumpSparseData(mesh, geofac_rot, level, edgesPerVertex,
                   std::string("laplICONmylib_geofacRot.txt"));
    dumpSparseData(mesh, geofac_div, level, edgesPerCell,
                   std::string("laplICONmylib_geofacDiv.txt"));
  }

  //===------------------------------------------------------------------------------------------===//
  // computation starts here
  //===------------------------------------------------------------------------------------------===//

  // SUBROUTINE rot_vertex_atmos
  for(const auto& v : mesh.vertices()) {
    int m_sparse = 0;
    for(const auto& e : v.edges()) {
      rot_vec(v, level) += vec(*e, level) * geofac_rot(v, m_sparse++, level);
    }
    if(v.vertices().size() != 6) {
      rot_vec(v, level) = 0.;
    }
  }

  // SUBROUTINE div
  for(const auto& c : mesh.faces()) {
    int m_sparse = 0;
    for(const auto& e : c.edges()) {
      div_vec(c, level) += vec(*e, level) * geofac_div(c, m_sparse++, level);
    }
  }

  if(dbg_out) {
    dumpField("laplICONmylib_div.txt", mesh, div_vec, level);
    dumpField("laplICONmylib_rot.txt", mesh, rot_vec, level);
  }

  // SUBROUTINE nabla2_vec
  for(const auto& e : mesh.edges()) {
    // ignore boundaries for now
    if(e.get().faces().size() == 1) {
      nabla2_vec(e, level) = 0.;
      continue;
    }

    if(e.get().vertex(1).vertices().size() != 6 || e.get().vertex(0).vertices().size() != 6) {
      nabla2_vec(e, level) = 0.;
      continue;
    }

    nabla2t1_vec(e, level) =
        tangent_orientation(e, level) *
        (rot_vec(e.get().vertex(1), level) - rot_vec(e.get().vertex(0), level)) /
        primal_edge_length(e.get(), level);

    nabla2t2_vec(e, level) = (div_vec(e.get().face(1), level) - div_vec(e.get().face(0), level)) /
                             dual_edge_length(e.get(), level);

    nabla2_vec(e, level) = nabla2t1_vec(e, level) - nabla2t2_vec(e, level);
  }

  if(dbg_out) {
    dumpField("laplICONmylib_rotH.txt", mesh, nabla2t1_vec, level, mylib::edge_color::horizontal);
    dumpField("laplICONmylib_rotV.txt", mesh, nabla2t1_vec, level, mylib::edge_color::vertical);
    dumpField("laplICONmylib_rotD.txt", mesh, nabla2t1_vec, level, mylib::edge_color::diagonal);

    dumpField("laplICONmylib_divH.txt", mesh, nabla2t2_vec, level, mylib::edge_color::horizontal);
    dumpField("laplICONmylib_divV.txt", mesh, nabla2t2_vec, level, mylib::edge_color::vertical);
    dumpField("laplICONmylib_divD.txt", mesh, nabla2t2_vec, level, mylib::edge_color::diagonal);
  }

  //===------------------------------------------------------------------------------------------===//
  // dumping a hopefully nice colorful laplacian
  //===------------------------------------------------------------------------------------------===//
  dumpField("laplICONmylib_out.txt", mesh, nabla2_vec, level);
}

void dumpMesh(const mylib::Grid& m, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(const auto& e : m.edges()) {
    fprintf(fp, "%f %f %f %f\n", e.get().vertex(0).x(), e.get().vertex(0).y(),
            e.get().vertex(1).x(), e.get().vertex(1).y());
  }
  fclose(fp);
}

void dumpDualMesh(const mylib::Grid& m, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(const auto& e : m.edges()) {
    if(e.get().faces().size() != 2) {
      continue;
    }

    // This is WRONG!, leads to a dual mesh which is not orthogonal to
    // primal mesh
    // auto [xm1, ym1] = CellMidPoint(e.get().face(0));
    // auto [xm2, ym2] = CellMidPoint(e.get().face(1));

    auto [xm1, ym1] = CellCircumcenter(e.get().face(0));
    auto [xm2, ym2] = CellCircumcenter(e.get().face(1));
    fprintf(fp, "%f %f %f %f\n", xm1, ym1, xm2, ym2);
  }
  fclose(fp);
}

void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseVertexData<double>& sparseData,
                    int level, int edgesPerVertex, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(const auto& v : mesh.vertices()) {
    int sparse_idx = 0;
    for(const auto& e : v.edges()) {
      double val = sparseData(v, sparse_idx, level);
      auto [emx, emy] = EdgeMidpoint(*e);
      double dx = emx - v.x();
      double dy = emy - v.y();
      fprintf(fp, "%f %f %f\n", v.x() + 0.5 * dx, v.y() + 0.5 * dy, val);
      sparse_idx++;
    }
  }
}

void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseFaceData<double>& sparseData,
                    int level, int edgesPerCell, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(const auto& c : mesh.faces()) {
    int sparse_idx = 0;
    auto [cmx, cmy] = CellCircumcenter(c);
    for(const auto& e : c.edges()) {
      double val = sparseData(c, sparse_idx, level);
      auto [emx, emy] = EdgeMidpoint(*e);
      double dx = emx - cmx;
      double dy = emy - cmy;
      fprintf(fp, "%f %f %f\n", cmx + 0.5 * dx, cmy + 0.5 * dy, val);
      sparse_idx++;
    }
  }
}

void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::EdgeData<double>& field, int level,
               std::optional<mylib::edge_color> color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& e : mesh.edges()) {
    if(color.has_value() && e.get().color() != color.value()) {
      continue;
    }
    auto [x, y] = EdgeMidpoint(e);
    fprintf(fp, "%f %f %f\n", x, y, field(e, level));
  }
  fclose(fp);
}

void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::EdgeData<double>& field_x, const mylib::EdgeData<double>& field_y,
               int level, std::optional<mylib::edge_color> color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& e : mesh.edges()) {
    if(color.has_value() && e.get().color() != color.value()) {
      continue;
    }
    auto [x, y] = EdgeMidpoint(e);
    fprintf(fp, "%f %f %f %f\n", x, y, field_x(e, level), field_y(e, level));
  }
  fclose(fp);
}

void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::FaceData<double>& field, int level,
               std::optional<mylib::face_color> color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& c : mesh.faces()) {
    if(color.has_value() && c.color() != color.value()) {
      continue;
    }
    auto [x, y] = CellCircumcenter(c);
    fprintf(fp, "%f %f %f\n", x, y, field(c, level));
  }
  fclose(fp);
}

void dumpField(const std::string& fname, const mylib::Grid& mesh,
               const mylib::VertexData<double>& field, int level) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& v : mesh.vertices()) {
    fprintf(fp, "%f %f %f\n", v.x(), v.y(), field(v, level));
  }
  fclose(fp);
}

// FILE* fpin = fopen("laplICONmylib_in_recons.txt", "w+");
// for(const auto& c : mesh.faces()) {
//   double inx = primal_normal_x(c.edge(0), level) * vec(c.edge(0), level) +
//                primal_normal_x(c.edge(1), level) * vec(c.edge(1), level) +
//                primal_normal_x(c.edge(2), level) * vec(c.edge(2), level);
//   double iny = primal_normal_y(c.edge(0), level) * vec(c.edge(0), level) +
//                primal_normal_y(c.edge(1), level) * vec(c.edge(1), level) +
//                primal_normal_y(c.edge(2), level) * vec(c.edge(2), level);
//   auto [xm, ym] = CellCircumcenter(c);
//   double l = sqrt(inx * inx + iny * iny);
//   // assert(iny < 1e3 * std::numeric_limits<double>::epsilon());
//   fprintf(fpin, "%f %f %f %f %f %f\n", xm, ym, inx, iny, inx / l, iny / l);
// }
