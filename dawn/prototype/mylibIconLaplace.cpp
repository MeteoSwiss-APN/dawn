#include "mylib.hpp"
#include "mylib_interface.hpp"
#include <assert.h>
#include <fstream>

#include "generated_triGradient.hpp"

// re-implemenation of the ICON nabla2_vec operator.
//    see mo_math_divrot.f90 and mo_math_laplace.f90
// names have been kept close to the FORTRAN code, but the "_Location" suffixes have been removed
// because of the strong typing in C++ and inconsistent application in the FORTRAN source

template <typename T>
int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

void dumpEdgeField(const std::string& fname, const mylib::Grid& mesh,
                   const mylib::EdgeData<double>& field, int level, mylib::edge_color color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& e : mesh.edges()) {
    if(e.get().color() != color) {
      continue;
    }
    double x = 0.;
    double y = 0.;
    int num_vertices = 0;
    for(auto& v : e.get().vertices()) {
      x += v->x();
      y += v->y();
      num_vertices++;
    }
    x /= num_vertices;
    y /= num_vertices;
    fprintf(fp, "%f %f %f\n", x, y, field(e, level));
  }
  fclose(fp);
}

void dumpEdgeField(const std::string& fname, const mylib::Grid& mesh,
                   const mylib::EdgeData<double>& field, int level) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(auto& e : mesh.edges()) {
    double x = 0.;
    double y = 0.;
    int num_vertices = 0;
    for(auto& v : e.get().vertices()) {
      x += v->x();
      y += v->y();
      num_vertices++;
    }
    x /= num_vertices;
    y /= num_vertices;
    fprintf(fp, "%f %f %f\n", x, y, field(e, level));
  }
  fclose(fp);
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
  std::vector<const mylib::Vertex*> cyclicVertices = center.vertices();
  cyclicVertices.push_back(&center.vertex(0));
  double totalArea = 0.;
  for(int i = 0; i < cyclicVertices.size() - 1; i++) {
    mylib::Vertex left = *cyclicVertices[i];
    mylib::Vertex right = *cyclicVertices[i + 1];

    mylib::Vertex leftHalf(center.x() + 0.5 * (left.x() - center.x()),
                           center.y() + 0.5 * (left.y() - center.y()), -1);
    mylib::Vertex rightHalf(center.x() + 0.5 * (right.x() - center.x()),
                            center.y() + 0.5 * (right.y() - center.y()), -1);
    totalArea += TriangleArea(center, leftHalf, rightHalf);
  }
  return totalArea;
}

double DualEdgeLength(const mylib::Edge& e) {
  if(e.faces().size() == 1) { // dual edge length is zero on boundaries!
    return 0.;
  }
  auto c0 = e.face(0);
  auto c1 = e.face(1);
  auto [x0, y0] = CellMidPoint(c0);
  auto [x1, y1] = CellMidPoint(c1);
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
  auto [x0, y0] = CellMidPoint(c0);
  auto [x1, y1] = CellMidPoint(c1);
  double c2c1x = x1 - x0;
  double c2c1y = y1 - y0;

  auto v0 = e.vertex(0);
  auto v1 = e.vertex(1);
  double v2v1x = v1.x() - v0.x();
  double v2v1y = v1.y() - v0.y();

  return sgn(c2c1x * v2v1y - c2c1y * v2v1x);
}

void dumpMesh(const mylib::Grid& m) {
  FILE* fp = fopen("mesh.txt", "w+");
  for(const auto& e : m.edges()) {
    fprintf(fp, "%f %f %f %f\n", e.get().vertex(0).x(), e.get().vertex(0).y(),
            e.get().vertex(1).x(), e.get().vertex(1).y());
  }
  fclose(fp);
}

void dumpDualMesh(const mylib::Grid& m) {
  FILE* fp = fopen("dualMesh.txt", "w+");
  for(const auto& e : m.edges()) {
    if(e.get().faces().size() != 2) {
      continue;
    }
    auto [xm1, ym1] = CellMidPoint(e.get().face(0));
    auto [xm2, ym2] = CellMidPoint(e.get().face(1));
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
    auto [cmx, cmy] = CellMidPoint(c);
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

int main() {
  int w = 10;
  int k_size = 1;
  const int level = 0;
  mylib::Grid mesh{w, w, false, M_PI, M_PI};
  dumpMesh(mesh);
  dumpDualMesh(mesh);

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
  // output field (field we want to take the laplacian of)
  //===------------------------------------------------------------------------------------------===//
  mylib::EdgeData<double> nabla2_vec(mesh, k_size);
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
  // fields containing geometric infromation
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

  // init zero and test function
  FILE* fp = fopen("laplICONmylib_in.txt", "w+");
  for(const auto& e : mesh.edges()) {
    auto [xm, ym] = EdgeMidpoint(e);
    vec(e.get(), level) = sin(xm) * sin(ym);
    nabla2_vec(e.get(), level) = 0;
    fprintf(fp, "%f %f %f\n", xm, ym, sin(xm) * sin(ym));
  }
  fclose(fp);

  for(const auto& v : mesh.vertices()) {
    rot_vec(v, level) = 0;
  }

  for(const auto& c : mesh.faces()) {
    div_vec(c, level) = 0;
  }

  // init geometric info for edges
  FILE* fpprimal = fopen("laplICONmylib_EdgeLength.txt", "w+");
  FILE* fpdual = fopen("laplICONmylib_dualEdgeLength.txt", "w+");
  FILE* fpnormal = fopen("laplICONmylib_nrm.txt", "w+");
  FILE* fpdnormal = fopen("laplICONmylib_dnrm.txt", "w+");
  for(auto const& e : mesh.edges()) {
    primal_edge_length(e, level) = EdgeLength(e);
    dual_edge_length(e, level) = DualEdgeLength(e);
    tangent_orientation(e, level) = TangentOrientation(e);
    auto [xm, ym] = EdgeMidpoint(e);
    fprintf(fpprimal, "%f %f %f\n", xm, ym, primal_edge_length(e, level));
    fprintf(fpdual, "%f %f %f\n", xm, ym, dual_edge_length(e, level));
    auto [nx, ny] = PrimalNormal(e);
    primal_normal_x(e, level) = nx;
    primal_normal_y(e, level) = ny;
    // The primal normal, dual normal
    // forms a left-handed coordinate system
    dual_normal_x(e, level) = ny;
    dual_normal_y(e, level) = -nx;

    fprintf(fpnormal, "%f %f %f %f\n", xm, ym, nx, ny);
    fprintf(fpdnormal, "%f %f %f %f\n", xm, ym, ny, -nx);
  }
  fclose(fpprimal);
  fclose(fpdual);
  fclose(fpnormal);
  fclose(fpdnormal);

  // init geometric info for cells
  for(const auto& c : mesh.faces()) {
    cell_area(c, level) = CellArea(c);
  }

  // init geometric info for vertices
  for(const auto& v : mesh.vertices()) {
    dual_cell_area(v, level) = DualCellArea(v);
  }

  // init edge orientations for vertices and cells
  auto dot = [](const mylib::Vertex& v1, const mylib::Vertex& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y();
  };

  // for(const auto& v : mesh.vertices()) {
  //   if(v.vertices().size() != 6) {
  //     continue;
  //   }
  //   for(int i = 0; i < 6; i++) {
  //     double dxV1 = v.vertex(i).x();
  //     double dyV1 = v.vertex(i).y();

  //     double dxV2 = v.x();
  //     double dyV2 = v.y();

  //     double dxV = v.x() - v.vertex(i).x();
  //     double dyV = v.y() - v.vertex(i).y();
  //     double lV = sqrt(dxV * dxV + dyV * dyV);

  //     double dxE1 = v.edge(i).vertex(1).x();
  //     double dyE1 = v.edge(i).vertex(1).y();

  //     double dxE2 = v.edge(i).vertex(0).x();
  //     double dyE2 = v.edge(i).vertex(0).y();

  //     double dxE = v.edge(i).vertex(1).x() - v.edge(i).vertex(0).x();
  //     double dyE = v.edge(i).vertex(1).y() - v.edge(i).vertex(0).y();
  //     double lE = sqrt(dxE * dxE + dyE * dyE);

  //     mylib::Vertex testV =
  //         mylib::Vertex((v.vertex(i).x() - v.x()) / lV, (v.vertex(i).y() - v.y()) / lV, -1);
  //     mylib::Vertex testE =
  //         mylib::Vertex((v.edge(i).vertex(0).x() - v.edge(i).vertex(1).x()) / lE,
  //                       (v.edge(i).vertex(0).y() - v.edge(i).vertex(1).y()) / lE, -1);

  //     double d = dot(testV, testE);
  //     assert(fabs(d - 1.) < 1e3 * std::numeric_limits<double>::epsilon());
  //   }
  // }

  //   ! fill the orientation
  // IF (patch%edges%vertex_idx(edge_index, edge_block, 1) == vertex_indexes(1) .AND. &
  //   & patch%edges%vertex_blk(edge_index, edge_block, 1) == vertex_blocks(1)  .AND. &
  //   & patch%edges%vertex_idx(edge_index, edge_block, 2) == vertex_indexes(2) .AND. &
  //   & patch%edges%vertex_blk(edge_index, edge_block, 2) == vertex_blocks(2)) THEN
  //   edge_orientation = patch%edges%tangent_orientation(edge_index, edge_block)
  // ELSEIF(patch%edges%vertex_idx(edge_index, edge_block, 1) == vertex_indexes(2) .AND. &
  //   & patch%edges%vertex_blk(edge_index, edge_block, 1) == vertex_blocks(2)  .AND. &
  //   & patch%edges%vertex_idx(edge_index, edge_block, 2) == vertex_indexes(1) .AND. &
  //   & patch%edges%vertex_blk(edge_index, edge_block, 2) == vertex_blocks(1)) THEN
  //   edge_orientation = -patch%edges%tangent_orientation(edge_index, edge_block)
  // ELSE

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
      edge_orientation_vertex(v, m_sparse, level) =
          sgn(dot(mylib::Vertex(v.vertex(m_sparse).x() - v.x(), v.vertex(m_sparse).y() - v.y(), -1),
                  mylib::Vertex(dual_normal_x(*e, level), dual_normal_y(*e, level), -1)));
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
    auto [xm, ym] = CellMidPoint(c);
    for(const auto& e : c.edges()) {
      mylib::Vertex vOutside(e->vertex(0).x() - xm, e->vertex(0).y() - ym, -1);
      edge_orientation_cell(c, m_sparse, level) =
          sgn(dot(mylib::Vertex(e->vertex(0).x() - xm, e->vertex(0).y() - ym, -1),
                  mylib::Vertex(tangent_orientation(*e, level) * primal_normal_x(*e, level),
                                tangent_orientation(*e, level) * primal_normal_y(*e, level), -1)));
      m_sparse++;
      // explanation: the vector cellModpoint -> e.vertex(0) is guaranteed to point outside. The dot
      // product checks if the edge normal has the same orientation. e.vertex(0) is arbitrary,
      // vertex(1), or any point on e would work just as well
    }
  }

  // init sparse quantities for div and rot
  for(const auto& v : mesh.vertices()) {
    int m_sparse = 0;
    for(const auto& e : v.edges()) {
      geofac_rot(v, m_sparse, level) = dual_edge_length(*e, level) *
                                       edge_orientation_vertex(v, m_sparse, level) *
                                       dual_cell_area(v, level);

      // ptr_int%geofac_rot(jv,je,jb) =                &
      //    & ptr_patch%edges%dual_edge_length(ile,ibe) * &
      //    & ptr_patch%verts%edge_orientation(jv,jb,je)/ &
      //    & ptr_patch%verts%dual_area(jv,jb) * REAL(ifac,wp)
      m_sparse++;
    }
  }

  dumpSparseData(mesh, geofac_rot, level, edgesPerVertex, std::string("geofac_rot.txt"));

  for(const auto& c : mesh.faces()) {
    int m_sparse = 0;
    for(const auto& e : c.edges()) {
      geofac_div(c, m_sparse, level) = primal_edge_length(*e, level) *
                                       edge_orientation_cell(c, m_sparse, level) *
                                       cell_area(c, level);

      //  ptr_int%geofac_div(jc,je,jb) = &
      //    & ptr_patch%edges%primal_edge_length(ile,ibe) * &
      //    & ptr_patch%cells%edge_orientation(jc,jb,je)  / &
      //    & ptr_patch%cells%area(jc,jb)
      m_sparse++;
    }
  }

  dumpSparseData(mesh, geofac_div, level, edgesPerCell, std::string("geofac_div.txt"));

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

  // FD like test
  // for(const auto& c : mesh.faces()) {
  //   auto [x, y] = CellMidPoint(c);
  //   div_vec(c, level) = sin(x) * sin(y) * (c.color() == mylib::face_color::upward ? 1 : -1);
  // }

  // for(const auto& v : mesh.vertices()) {
  //   rot_vec(v, level) = sin(v.x()) * sin(v.y());
  // }

  FILE* fpdiv = fopen("laplICONmylib_div.txt", "w+");
  FILE* fprot = fopen("laplICONmylib_rot.txt", "w+");

  FILE* fpdivH = fopen("laplICONmylib_divH.txt", "w+");
  FILE* fprotH = fopen("laplICONmylib_rotH.txt", "w+");
  FILE* fpdivV = fopen("laplICONmylib_divV.txt", "w+");
  FILE* fprotV = fopen("laplICONmylib_rotV.txt", "w+");
  FILE* fpdivD = fopen("laplICONmylib_divD.txt", "w+");
  FILE* fprotD = fopen("laplICONmylib_rotD.txt", "w+");

  for(const auto& v : mesh.vertices()) {
    fprintf(fprot, "%f %f %f\n", v.x(), v.y(), rot_vec(v, level));
  }

  for(const auto& c : mesh.faces()) {
    auto [xm, ym] = CellMidPoint(c);
    fprintf(fpdiv, "%f %f %f\n", xm, ym, div_vec(c, level));
  }

  // SUBROUTINE nabla2_vec
  for(const auto& e : mesh.edges()) {
    auto [x, y] = EdgeMidpoint(e.get());
    if(e.get().faces().size() == 1) { // ignore boundaries for now
      nabla2_vec(e, level) = 0.;
      continue;
    }

    if(e.get().color() == mylib::edge_color::horizontal) {
      fprintf(fpdivH, "%f %f %f\n", x, y,
              (div_vec(e.get().face(1), level) - div_vec(e.get().face(0), level)) /
                  dual_edge_length(e, level));

      fprintf(fprotH, "%f %f %f\n", x, y,
              (tangent_orientation(e, level) *
               (rot_vec(e.get().vertex(1), level) - rot_vec(e.get().vertex(0), level)) /
               primal_edge_length(e.get(), level)));
    }

    if(e.get().color() == mylib::edge_color::vertical) {
      fprintf(fpdivV, "%f %f %f\n", x, y,
              (div_vec(e.get().face(1), level) - div_vec(e.get().face(0), level)) /
                  dual_edge_length(e, level));

      fprintf(fprotV, "%f %f %f\n", x, y,
              (tangent_orientation(e, level) *
               (rot_vec(e.get().vertex(1), level) - rot_vec(e.get().vertex(0), level)) /
               primal_edge_length(e.get(), level)));
    }

    if(e.get().color() == mylib::edge_color::diagonal) {
      fprintf(fpdivD, "%f %f %f\n", x, y,
              (div_vec(e.get().face(1), level) - div_vec(e.get().face(0), level)) /
                  dual_edge_length(e, level));

      fprintf(fprotD, "%f %f %f\n", x, y,
              (tangent_orientation(e, level) *
               (rot_vec(e.get().vertex(1), level) - rot_vec(e.get().vertex(0), level)) /
               primal_edge_length(e.get(), level)));
    }

    // the second term here is conceptually quite surprising. there is no correction term w.r.t to
    // the orientation of face(1) and face(0) w.r.t the edge, which implies that the faces are well
    // ordered from the perspective of the edges in ICON or the cell values are pre-mutiplied with
    // the appropriate sign. probably the latter! EDIT: this seems to be the case
    nabla2_vec(e, level) +=
        tangent_orientation(e, level) *
            (rot_vec(e.get().vertex(1), level) - rot_vec(e.get().vertex(0), level)) /
            primal_edge_length(e.get(), level) +
        (div_vec(e.get().face(1), level) - div_vec(e.get().face(0), level)) /
            dual_edge_length(e.get(), level);
  }
  fclose(fpdiv);
  fclose(fprot);
  fclose(fpdivH);
  fclose(fprotH);
  fclose(fpdivV);
  fclose(fprotV);
  fclose(fpdivD);
  fclose(fprotD);

  //===------------------------------------------------------------------------------------------===//
  // dumping a hopefully nice colorful laplacian
  //===------------------------------------------------------------------------------------------===//
  dumpEdgeField("laplICONmylib_outD.txt", mesh, nabla2_vec, level, mylib::edge_color::diagonal);
  dumpEdgeField("laplICONmylib_outH.txt", mesh, nabla2_vec, level, mylib::edge_color::horizontal);
  dumpEdgeField("laplICONmylib_outV.txt", mesh, nabla2_vec, level, mylib::edge_color::vertical);

  dumpEdgeField("laplICONmylib_out.txt", mesh, nabla2_vec, level);
}
