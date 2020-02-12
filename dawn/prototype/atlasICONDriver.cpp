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

//===------------------------------------------------------------------------------------------===//
//
//
//    WARNING! THIS IS A PROTOTYPE! DOES NOT YET PRODUCE CORRECT RESULTS! WARNING!
//
//
//===------------------------------------------------------------------------------------------===//

#include <cmath>
#include <vector>

#include "AtlasCartesianWrapper.h"
#include "atlas/grid/Grid.h"
#include "atlas/grid/Iterator.h"
#include "atlas/grid/StructuredGrid.h"
#include "atlas/grid/detail/grid/Structured.h"
#include "atlas/library/Library.h"
#include "atlas/library/config.h"
#include "atlas/mesh/Mesh.h"
#include "atlas/mesh/actions/BuildEdges.h"
#include "atlas/meshgenerator.h"

#include "atlas/functionspace/CellColumns.h"
#include "atlas/functionspace/EdgeColumns.h"
#include "atlas/functionspace/NodeColumns.h"

#include "atlas_interface.hpp"

#include "generated_iconLaplace.hpp"

// remove later
#include "atlas/output/Gmsh.h"

template <typename T>
static int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

atlas::Mesh makeAtlasMesh(int nx, double L) {
  atlas::Grid grid;

  // Create grid

  // this is adapted from
  // https://github.com/ecmwf/atlas/blob/a0017406f7ae54d306c9585113201af18d86fa40/src/tests/grid/test_grids.cc#L352
  //
  //    here, the grid is simple right triangles with strict up/down orientation. a transform will
  //    be applied later using the AtlasToCartesian wrapper to make the tris equilateral
  {
    using XSpace = atlas::StructuredGrid::XSpace;
    using YSpace = atlas::StructuredGrid::YSpace;
    auto xspace = atlas::util::Config{};
    xspace.set("type", "linear");
    xspace.set("N", nx + 1);
    xspace.set("length", L);
    xspace.set("endpoint", false);
    xspace.set("start[]", std::vector<double>(nx + 1, 0));
    grid =
        atlas::StructuredGrid{XSpace{xspace}, YSpace{atlas::grid::LinearSpacing{{0., L}, nx + 1}}};
  }

  auto meshgen = atlas::StructuredMeshGenerator{atlas::util::Config("angle", -1.)};
  return meshgen.generate(grid);
}

//===------------------------------------------------------------------------------------------===//
// output (debugging)
//===------------------------------------------------------------------------------------------===//
void dumpMesh(const atlas::Mesh& m, AtlasToCartesian& wrapper, const std::string& fname);
void dumpDualMesh(const atlas::Mesh& m, AtlasToCartesian& wrapper, const std::string& fname);

void dumpNodeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level);
void dumpCellField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level);
void dumpEdgeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level,
                   std::optional<Orientation> color = std::nullopt);
void dumpEdgeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field_x, atlasInterface::Field<double>& field_y,
                   int level, std::optional<Orientation> color = std::nullopt);

int main() {
  int w = 15;
  int k_size = 1;
  const int level = 0;
  double lDomain = M_PI;

  const bool dbg_out = true;

  atlas::Mesh mesh = makeAtlasMesh(w, lDomain);
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  const bool skewToEquilateral = true;
  AtlasToCartesian wrapper(mesh, lDomain, skewToEquilateral);

  if(dbg_out) {
    dumpMesh(mesh, wrapper, "laplICONatlas_mesh.txt");
    dumpDualMesh(mesh, wrapper, "laplICONatlas_dualMesh.txt");
  }

  const int edgesPerVertex = 6;
  const int edgesPerCell = 3;

  atlas::functionspace::CellColumns fs_cells(mesh, atlas::option::levels(k_size));
  atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(k_size));
  atlas::functionspace::EdgeColumns fs_edges(mesh, atlas::option::levels(k_size));

  //===------------------------------------------------------------------------------------------===//
  // input field (field we want to take the laplacian of)
  //===------------------------------------------------------------------------------------------===//
  atlas::Field vec_F{fs_edges.createField<double>(atlas::option::name("vec"))};
  atlasInterface::Field<double> vec = atlas::array::make_view<double, 2>(vec_F);
  //    this is, confusingly, called vec_e, even though it is a scalar
  //    _conceptually_, this can be regarded as a vector with implicit direction (presumably normal
  //    to edge direction)

  //===------------------------------------------------------------------------------------------===//
  // output field (field containing the computed laplacian)
  //===------------------------------------------------------------------------------------------===//
  atlas::Field nabla2_vec_F{fs_edges.createField<double>(atlas::option::name("nabla2_vec"))};
  atlasInterface::Field<double> nabla2_vec = atlas::array::make_view<double, 2>(nabla2_vec_F);
  atlas::Field nabla2t1_vec_F{fs_edges.createField<double>(
      atlas::option::name("nabla2t1_vec"))}; // term 1 and term 2 of nabla for debugging
  atlasInterface::Field<double> nabla2t1_vec = atlas::array::make_view<double, 2>(nabla2t1_vec_F);
  atlas::Field nabla2t2_vec_F{fs_edges.createField<double>(atlas::option::name("nabla2t2_vec"))};
  atlasInterface::Field<double> nabla2t2_vec = atlas::array::make_view<double, 2>(nabla2t2_vec_F);
  //    again, surprisingly enough, this is a scalar quantity even though the vector laplacian is
  //    a laplacian.

  //===------------------------------------------------------------------------------------------===//
  // intermediary fields (curl/rot and div of vec_e)
  //===------------------------------------------------------------------------------------------===//

  // rotation (more commonly curl) of vec_e on vertices
  //    I'm not entirely positive how one can take the curl of a scalar field (commonly a undefined
  //    operation), however, since vec_e is _conceptually_ a vector this works out. somehow.
  atlas::Field rot_vec_F{fs_nodes.createField<double>(atlas::option::name("rot_vec"))};
  atlasInterface::Field<double> rot_vec = atlas::array::make_view<double, 2>(rot_vec_F);

  // divergence of vec_e on cells
  //    Again, not entirely sure how one can measure the divergence of scalars, but again, vec_e is
  //    _conceptually_ a vector, so...
  atlas::Field div_vec_F{fs_cells.createField<double>(atlas::option::name("div_vec"))};
  atlasInterface::Field<double> div_vec = atlas::array::make_view<double, 2>(div_vec_F);

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

  atlas::Field geofac_rot_F{fs_nodes.createField<double>(atlas::option::name("geofac_rot") |
                                                         atlas::option::variables(edgesPerVertex))};
  atlasInterface::SparseDimension<double> geofac_rot =
      atlas::array::make_view<double, 3>(geofac_rot_F);
  atlas::Field edge_orientation_vertex_F{fs_nodes.createField<double>(
      atlas::option::name("edge_orientation_vertex") | atlas::option::variables(edgesPerVertex))};
  atlasInterface::SparseDimension<double> edge_orientation_vertex =
      atlas::array::make_view<double, 3>(edge_orientation_vertex_F);

  // needed for the computation of the curl/rotation. according to documentation this needs to be:
  //   ! ...the appropriate cell based edge_orientation is required to
  //   ! obtain the correct value for the application of Gauss theorem
  //   ! (which requires the scalar product of the vector field with the
  //   ! OUTWARD pointing unit vector with respect to cell jc; since the
  //   ! positive direction for the vector components is not necessarily
  //   ! the outward pointing one with respect to cell jc, a correction
  //   ! coefficient (equal to +-1) is necessary, given by
  //   ! ptr_patch%grid%cells%edge_orientation)

  atlas::Field geofac_div_F{fs_cells.createField<double>(atlas::option::name("geofac_div") |
                                                         atlas::option::variables(edgesPerCell))};
  atlasInterface::SparseDimension<double> geofac_div =
      atlas::array::make_view<double, 3>(geofac_div_F);
  atlas::Field edge_orientation_cell_F{fs_cells.createField<double>(
      atlas::option::name("edge_orientation_cell") | atlas::option::variables(edgesPerCell))};
  atlasInterface::SparseDimension<double> edge_orientation_cell =
      atlas::array::make_view<double, 3>(edge_orientation_cell_F);

  // //===------------------------------------------------------------------------------------------===//
  // // fields containing geometric information
  // //===------------------------------------------------------------------------------------------===//
  atlas::Field tangent_orientation_F{
      fs_edges.createField<double>(atlas::option::name("tangent_orientation"))};
  atlasInterface::Field<double> tangent_orientation =
      atlas::array::make_view<double, 2>(tangent_orientation_F);
  atlas::Field primal_edge_length_F{
      fs_edges.createField<double>(atlas::option::name("primal_edge_length"))};
  atlasInterface::Field<double> primal_edge_length =
      atlas::array::make_view<double, 2>(primal_edge_length_F);
  atlas::Field dual_edge_length_F{
      fs_edges.createField<double>(atlas::option::name("dual_edge_length"))};
  atlasInterface::Field<double> dual_edge_length =
      atlas::array::make_view<double, 2>(dual_edge_length_F);
  atlas::Field dual_normal_x_F{fs_edges.createField<double>(atlas::option::name("dual_normal_x"))};
  atlasInterface::Field<double> dual_normal_x = atlas::array::make_view<double, 2>(dual_normal_x_F);
  atlas::Field dual_normal_y_F{fs_edges.createField<double>(atlas::option::name("dual_normal_y"))};
  atlasInterface::Field<double> dual_normal_y = atlas::array::make_view<double, 2>(dual_normal_y_F);
  atlas::Field primal_normal_x_F{
      fs_edges.createField<double>(atlas::option::name("primal_normal_x"))};
  atlasInterface::Field<double> primal_normal_x =
      atlas::array::make_view<double, 2>(primal_normal_x_F);
  atlas::Field primal_normal_y_F{
      fs_edges.createField<double>(atlas::option::name("primal_normal_y"))};
  atlasInterface::Field<double> primal_normal_y =
      atlas::array::make_view<double, 2>(primal_normal_y_F);

  atlas::Field cell_area_F{fs_cells.createField<double>(atlas::option::name("cell_area"))};
  atlasInterface::Field<double> cell_area = atlas::array::make_view<double, 2>(cell_area_F);

  atlas::Field dual_cell_area_F{
      fs_nodes.createField<double>(atlas::option::name("dual_cell_area"))};
  atlasInterface::Field<double> dual_cell_area =
      atlas::array::make_view<double, 2>(dual_cell_area_F);

  //===------------------------------------------------------------------------------------------===//
  // initialize fields
  //===------------------------------------------------------------------------------------------===//

  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    rot_vec(nodeIdx, level) = 0;
  }

  for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
    div_vec(cellIdx, level) = 0;
  }

  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    primal_edge_length(edgeIdx, level) = wrapper.edgeLength(mesh, edgeIdx);
    dual_edge_length(edgeIdx, level) = wrapper.dualEdgeLength(mesh, edgeIdx);
    tangent_orientation(edgeIdx, level) = wrapper.tangentOrientation(mesh, edgeIdx);
    auto [nx, ny] = wrapper.primalNormal(mesh, edgeIdx);
    primal_normal_x(edgeIdx, level) = nx * tangent_orientation(edgeIdx, level);
    primal_normal_y(edgeIdx, level) = ny * tangent_orientation(edgeIdx, level);
    // The primal normal, dual normal
    // forms a left-handed coordinate system
    dual_normal_x(edgeIdx, level) = ny;
    dual_normal_y(edgeIdx, level) = -nx;
  }

  if(dbg_out) {
    dumpEdgeField("laplICONatlas_EdgeLength.txt", mesh, wrapper, primal_edge_length, level);
    dumpEdgeField("laplICONatlas_dualEdgeLength.txt", mesh, wrapper, dual_edge_length, level);
    dumpEdgeField("laplICONatlas_nrm.txt", mesh, wrapper, primal_normal_x, primal_normal_y, level);
    dumpEdgeField("laplICONatlas_dnrm.txt", mesh, wrapper, dual_normal_x, dual_normal_y, level);
  }

  auto wave = [](double px, double py) { return sin(px) * sin(py); };
  auto constant = [](double px, double py) { return 1.; };
  auto lin = [](double px, double py) { return px; };

  // init zero and test function
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
    double py = 2 / sqrt(3) * ym;
    double px = xm + 0.5 * py;

    // this way to initialize the field is wrong, or at least does it does not correspond to what
    // one might expect intuitively. the values on the edges are the lengths of vectors in the
    // direction of the edge normal. assigning a constant field would thus mean that quantity 1
    // flows into the cell on two edges, and out on another (or vice versa). Divergence will hence
    // not be zero in this case!
    double fun = wave(px, py);
    vec(edgeIdx, level) = fun;

    nabla2_vec(edgeIdx, level) = 0;
  }

  // init geometric info for cells
  for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
    cell_area(cellIdx, level) = wrapper.cellArea(mesh, cellIdx);
  }
  // init geometric info for vertices
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    dual_cell_area(nodeIdx, level) = wrapper.dualCellArea(mesh, nodeIdx);
  }

  if(dbg_out) {
    dumpCellField("laplICONatlas_areaCell.txt", mesh, wrapper, cell_area, level);
    dumpNodeField("laplICONatlas_areaCellDual.txt", mesh, wrapper, dual_cell_area, level);
  }

  // init edge orientations for vertices and cells
  auto dot = [](const Vector& v1, const Vector& v2) {
    return std::get<0>(v1) * std::get<0>(v2) + std::get<1>(v1) * std::get<1>(v2);
  };

  // +1 when the vector from this to the neigh-
  // bor vertex has the same orientation as the
  // tangent unit vector of the connecting edge.
  // -1 otherwise

  auto nodeNeighboursOfNode = [](atlas::Mesh const& m, int const& idx) {
    const auto& conn_nodes_to_edge = m.nodes().edge_connectivity();
    auto neighs = std::vector<std::tuple<int, int>>{};
    for(int ne = 0; ne < conn_nodes_to_edge.cols(idx); ++ne) {
      int nbh_edge_idx = conn_nodes_to_edge(idx, ne);
      const auto& conn_edge_to_nodes = m.edges().node_connectivity();
      for(int nn = 0; nn < conn_edge_to_nodes.cols(nbh_edge_idx); ++nn) {
        int nbhNode = conn_edge_to_nodes(idx, nn);
        if(nbhNode != idx) {
          neighs.emplace_back(std::tuple<int, int>(nbh_edge_idx, nbhNode));
        }
      }
    }
    return neighs;
  };

  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    const atlas::mesh::Nodes::Connectivity& nodeEdgeConnectivity = mesh.nodes().edge_connectivity();
    const atlas::mesh::HybridElements::Connectivity& edgeNodeConnectivity =
        mesh.edges().node_connectivity();

    const int missingVal = nodeEdgeConnectivity.missing_value();
    int numNbh = nodeEdgeConnectivity.cols(nodeIdx);

    if(numNbh != 6) {
      continue;
    }

    auto nbh = nodeNeighboursOfNode(mesh, nodeIdx);
    auto [cx, cy] = wrapper.nodeLocation(nodeIdx);

    for(int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++) {
      int edgeIdx = std::get<0>(nbh[nbhIdx]);
      int farNodeIdx = std::get<1>(nbh[nbhIdx]);

      int nodeIdxLo = edgeNodeConnectivity(edgeIdx, 0);
      int nodeIdxHi = edgeNodeConnectivity(edgeIdx, 1);

      auto [xLo, yLo] = wrapper.nodeLocation(nodeIdxLo);
      auto [xHi, yHi] = wrapper.nodeLocation(nodeIdxHi);

      auto [farX, farY] = wrapper.nodeLocation(farNodeIdx);

      Vector edgeVec{xHi - xLo, yHi - yLo};
      // its not quite clear how to implement this properly in Atlas
      //        nodes have no node neighbors on an atlas grid
      //        Vector dualNrm{cx - farX, cy - farY}; <- leads to oscillations in rot field
      Vector dualNrm{dual_normal_x(edgeIdx, level), dual_normal_y(edgeIdx, level)};
      edge_orientation_vertex(nodeIdx, nbhIdx, level) = sgn(dot(edgeVec, dualNrm));
    }
  }

  // The orientation of the edge normal vector
  // (the variable primal normal in the edges ta-
  // ble) for the cell according to Gauss formula.
  // It is equal to +1 if the normal to the edge
  // is outwards from the cell, otherwise is -1.
  for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
    const atlas::mesh::HybridElements::Connectivity& cellEdgeConnectivity =
        mesh.cells().edge_connectivity();
    auto [xm, ym] = wrapper.cellCircumcenter(mesh, cellIdx);

    const int missingVal = cellEdgeConnectivity.missing_value();
    int numNbh = cellEdgeConnectivity.cols(cellIdx);
    assert(numNbh == edgesPerCell);

    for(int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++) {
      int edgeIdx = cellEdgeConnectivity(cellIdx, nbhIdx);
      auto [emX, emY] = wrapper.edgeMidpoint(mesh, edgeIdx);
      Vector toOutsdie{emX - xm, emY - ym};
      Vector primal = {primal_normal_x(edgeIdx, level), primal_normal_y(edgeIdx, level)};
      edge_orientation_cell(cellIdx, nbhIdx, level) = sgn(dot(toOutsdie, primal));
    }
    // explanation: the vector cellMidpoint -> edgeMidpoint is guaranteed to point outside. The dot
    // product checks if the edge normal has the same orientation. edgeMidpoint is arbitrary, any
    // point on e would work just as well
  }

  // init sparse quantities for div and rot
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    const atlas::mesh::Nodes::Connectivity& nodeEdgeConnectivity = mesh.nodes().edge_connectivity();

    int numNbh = nodeEdgeConnectivity.cols(nodeIdx);
    // assert(numNbh == edgesPerVertex);

    for(int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++) {
      int edgeIdx = nodeEdgeConnectivity(nodeIdx, nbhIdx);
      geofac_rot(nodeIdx, nbhIdx, level) = dual_edge_length(edgeIdx, level) *
                                           edge_orientation_vertex(nodeIdx, nbhIdx, level) /
                                           dual_cell_area(nodeIdx, level);
    }
    // ptr_int%geofac_rot(jv,je,jb) =                &
    //    & ptr_patch%edges%dual_edge_length(ile,ibe) * &
    //    & ptr_patch%verts%edge_orientation(jv,jb,je)/ &
    //    & ptr_patch%verts%dual_area(jv,jb) * REAL(ifac,wp)
  }

  for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
    const atlas::mesh::HybridElements::Connectivity& cellEdgeConnectivity =
        mesh.cells().edge_connectivity();

    int numNbh = cellEdgeConnectivity.cols(cellIdx);
    assert(numNbh == edgesPerCell);

    for(int nbhIdx = 0; nbhIdx < numNbh; nbhIdx++) {
      int edgeIdx = cellEdgeConnectivity(cellIdx, nbhIdx);
      geofac_div(cellIdx, nbhIdx, level) = primal_edge_length(edgeIdx, level) *
                                           edge_orientation_cell(cellIdx, nbhIdx, level) /
                                           cell_area(cellIdx, level);
    }
    //  ptr_int%geofac_div(jc,je,jb) = &
    //    & ptr_patch%edges%primal_edge_length(ile,ibe) * &
    //    & ptr_patch%cells%edge_orientation(jc,jb,je)  / &
    //    & ptr_patch%cells%area(jc,jb)
  }

  //===------------------------------------------------------------------------------------------===//
  // stencil call
  //===------------------------------------------------------------------------------------------===//

  dawn_generated::cxxnaiveico::icon<atlasInterface::atlasTag>(
      mesh, k_size, vec, div_vec, rot_vec, nabla2t1_vec, nabla2t2_vec, nabla2_vec,
      primal_edge_length, dual_edge_length, tangent_orientation, geofac_rot, geofac_div)
      .run();

  if(dbg_out) {
    dumpCellField("laplICONatlas_div.txt", mesh, wrapper, div_vec, level);
    dumpNodeField("laplICONatlas_rot.txt", mesh, wrapper, rot_vec, level);

    dumpEdgeField("laplICONatlas_rotH.txt", mesh, wrapper, nabla2t1_vec, level,
                  Orientation::Horizontal);
    dumpEdgeField("laplICONatlas_rotV.txt", mesh, wrapper, nabla2t1_vec, level,
                  Orientation::Vertical);
    dumpEdgeField("laplICONatlas_rotD.txt", mesh, wrapper, nabla2t1_vec, level,
                  Orientation::Diagonal);

    dumpEdgeField("laplICONatlas_divH.txt", mesh, wrapper, nabla2t2_vec, level,
                  Orientation::Horizontal);
    dumpEdgeField("laplICONatlas_divV.txt", mesh, wrapper, nabla2t2_vec, level,
                  Orientation::Vertical);
    dumpEdgeField("laplICONatlas_divD.txt", mesh, wrapper, nabla2t2_vec, level,
                  Orientation::Diagonal);
  }

  //===------------------------------------------------------------------------------------------===//
  // dumping a hopefully nice colorful laplacian
  //===------------------------------------------------------------------------------------------===//
  dumpEdgeField("laplICONatlas_out.txt", mesh, wrapper, nabla2_vec, level);

  return 0;
}

void dumpMesh(const atlas::Mesh& mesh, AtlasToCartesian& wrapper, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  const atlas::mesh::HybridElements::Connectivity& edgeNodeConnectivity =
      mesh.edges().node_connectivity();
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    int numNbh = edgeNodeConnectivity.cols(edgeIdx);
    assert(numNbh == 2);

    int nbhLo = edgeNodeConnectivity(edgeIdx, 0);
    int nbhHi = edgeNodeConnectivity(edgeIdx, 1);

    auto [xLo, yLo] = wrapper.nodeLocation(nbhLo);
    auto [xHi, yHi] = wrapper.nodeLocation(nbhHi);

    fprintf(fp, "%f %f %f %f\n", xLo, yLo, xHi, yHi);
  }
  fclose(fp);
}

void dumpDualMesh(const atlas::Mesh& mesh, AtlasToCartesian& wrapper, const std::string& fname) {
  FILE* fp = fopen(fname.c_str(), "w+");
  const atlas::mesh::HybridElements::Connectivity& edgeCellConnectivity =
      mesh.edges().cell_connectivity();
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {

    int nbhLo = edgeCellConnectivity(edgeIdx, 0);
    int nbhHi = edgeCellConnectivity(edgeIdx, 1);

    if(nbhLo == edgeCellConnectivity.missing_value() ||
       nbhHi == edgeCellConnectivity.missing_value()) {
      continue;
    }

    auto [xm1, ym1] = wrapper.cellCircumcenter(mesh, nbhLo);
    auto [xm2, ym2] = wrapper.cellCircumcenter(mesh, nbhHi);

    fprintf(fp, "%f %f %f %f\n", xm1, ym1, xm2, ym2);
  }
  fclose(fp);
}

void dumpNodeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
    auto [xm, ym] = wrapper.nodeLocation(nodeIdx);
    fprintf(fp, "%f %f %f\n", xm, ym, field(nodeIdx, level));
  }
  fclose(fp);
}

void dumpCellField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
    auto [xm, ym] = wrapper.cellCircumcenter(mesh, cellIdx);
    fprintf(fp, "%f %f %f\n", xm, ym, field(cellIdx, level));
  }
  fclose(fp);
}

void dumpEdgeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field, int level,
                   std::optional<Orientation> color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    if(color.has_value() && wrapper.edgeOrientation(mesh, edgeIdx) != color.value()) {
      continue;
    }
    auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
    fprintf(fp, "%f %f %f\n", xm, ym, field(edgeIdx, level));
  }
  fclose(fp);
}

void dumpEdgeField(const std::string& fname, const atlas::Mesh& mesh, AtlasToCartesian wrapper,
                   atlasInterface::Field<double>& field_x, atlasInterface::Field<double>& field_y,
                   int level, std::optional<Orientation> color) {
  FILE* fp = fopen(fname.c_str(), "w+");
  for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
    if(color.has_value() && wrapper.edgeOrientation(mesh, edgeIdx) != color.value()) {
      continue;
    }
    auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
    fprintf(fp, "%f %f %f %f\n", xm, ym, field_x(edgeIdx, level), field_y(edgeIdx, level));
  }
  fclose(fp);
}

//   auto nodeNeighboursOfNode = [](atlas::Mesh const& m, int const& idx) {
//     const auto& conn_nodes_to_edge = m.nodes().edge_connectivity();
//     auto neighs = std::vector<std::tuple<int, int>>{};
//     for(int ne = 0; ne < conn_nodes_to_edge.cols(idx); ++ne) {
//       int nbh_edge_idx = conn_nodes_to_edge(idx, ne);
//       const auto& conn_edge_to_nodes = m.edges().node_connectivity();
//       for(int nn = 0; nn < conn_edge_to_nodes.cols(nbh_edge_idx); ++nn) {
//         int nbhNode = conn_edge_to_nodes(idx, nn);
//         if(nbhNode != idx) {
//           neighs.emplace_back(std::tuple<int, int>(nbh_edge_idx, nbhNode));
//         }
//       }
//     }
//     return neighs;
//   };

//   for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
//     const atlas::mesh::Nodes::Connectivity& nodeEdgeConnectivity =
//     mesh.nodes().edge_connectivity(); const atlas::mesh::HybridElements::Connectivity&
//     edgeNodeConnectivity =
//         mesh.edges().node_connectivity();

//     const int missingVal = nodeEdgeConnectivity.missing_value();
//     int numNbh = nodeEdgeConnectivity.cols(nodeIdx);

//     if(numNbh != 6) {
//       continue;
//     }

//     auto [vx, vy] = wrapper.nodeLocation(nodeIdx);
//     auto nodeNbh = nodeNeighboursOfNode(mesh, nodeIdx);

//     int nbh_idx = 0;
//     for(const auto& neighbor : nodeNbh) {
//       int edgeIdx = std::get<0>(neighbor);
//       int nodeIdx = std::get<1>(neighbor);

//       auto [farVx, farVy] = wrapper.nodeLocation(nodeIdx);

//       Vector testVector{farVx - vx, farVy - vy};

//       Vector dual = {dual_normal_x(edgeIdx, level), dual_normal_y(edgeIdx, level)};
//       edge_orientation_vertex(nodeIdx, nbh_idx, level) = sgn(dot(testVector, dual));
//       nbh_idx++;
//     }
//   }