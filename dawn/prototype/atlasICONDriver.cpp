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

// remove later
#include "atlas/output/Gmsh.h"

atlas::Mesh makeAtlasMesh(int nx, double L) {
  atlas::Grid grid;

  // Create grid

  // this is adapted from
  // https://github.com/ecmwf/atlas/blob/a0017406f7ae54d306c9585113201af18d86fa40/src/tests/grid/test_grids.cc#L352
  //
  //    here, the grid is simple right triangles with strict up/down orientation. a transform will
  //    be applied later to make the tris equilateral
  {
    using XSpace = atlas::StructuredGrid::XSpace;
    using YSpace = atlas::StructuredGrid::YSpace;
    auto xspace = atlas::util::Config{};
    xspace.set("type", "linear");
    xspace.set("N", nx);
    xspace.set("length", L);
    xspace.set("endpoint", false);
    xspace.set("start[]", std::vector<double>(nx, 0));
    grid = atlas::StructuredGrid{XSpace{xspace}, YSpace{atlas::grid::LinearSpacing{{0., L}, nx}}};
  }

  auto meshgen = atlas::StructuredMeshGenerator{atlas::util::Config("angle", -1.)};
  // auto mesh = meshgen.generate(grid);
  // auto gmsh = atlas::output::Gmsh{"structured_triangulated.msh"};
  // gmsh.write(mesh);
  // exit(0);

  return meshgen.generate(grid);
}

//===------------------------------------------------------------------------------------------===//
// output (debugging)
//===------------------------------------------------------------------------------------------===//
void dumpMesh(const atlas::Mesh& m, AtlasToCartesian& wrapper, const std::string& fname);
void dumpDualMesh(const atlas::Mesh& m, AtlasToCartesian& wrapper, const std::string& fname);

// void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseVertexData<double>& sparseData,
//                     int level, int edgesPerVertex, const std::string& fname);
// void dumpSparseData(const mylib::Grid& mesh, const mylib::SparseFaceData<double>& sparseData,
//                     int level, int edgesPerCell, const std::string& fname);

// void dumpField(const std::string& fname, const mylib::Grid& mesh,
//                const mylib::EdgeData<double>& field, int level,
//                std::optional<mylib::edge_color> color = std::nullopt);
// void dumpField(const std::string& fname, const mylib::Grid& mesh,
//                const mylib::EdgeData<double>& field_x, const mylib::EdgeData<double>& field_y,
//                int level, std::optional<mylib::edge_color> color = std::nullopt);
// void dumpField(const std::string& fname, const mylib::Grid& mesh,
//                const mylib::FaceData<double>& field, int level,
//                std::optional<mylib::face_color> color = std::nullopt);
// void dumpField(const std::string& fname, const mylib::Grid& mesh,
//                const mylib::VertexData<double>& field, int level);

int main() {
  int w = 20;
  int k_size = 1;
  const int level = 0;
  double lDomain = M_PI;

  const bool dbg_out = true;

  atlas::Mesh mesh = makeAtlasMesh(w, lDomain);
  atlas::mesh::actions::build_edges(mesh, atlas::util::Config("pole_edges", false));
  atlas::mesh::actions::build_node_to_edge_connectivity(mesh);
  atlas::mesh::actions::build_element_to_edge_connectivity(mesh);

  const bool skewToEquilateral = true;
  AtlasToCartesian wrapper(mesh, skewToEquilateral);

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
  // output field (field we want to take the laplacian of)
  //===------------------------------------------------------------------------------------------===//
  atlas::Field nabla2_vec_F{fs_edges.createField<double>(atlas::option::name("nabla2_vec"))};
  atlasInterface::Field<double> nabla2_vec = atlas::array::make_view<double, 2>(nabla2_vec_F);
  atlas::Field nabla2t1_vec_F{fs_edges.createField<double>(
      atlas::option::name("nabla2t1_vec"))}; // term 1 and term 2 of nabla for debugging
  atlasInterface::Field<double> nabla2t1_vec = atlas::array::make_view<double, 2>(nabla2t1_vec_F);
  atlas::Field nabla2t2_vec_F{fs_edges.createField<double>(atlas::option::name("nabla2t1_vec"))};
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
  // // fields containing geometric infromation
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

  // FILE* fp = fopen("dbg.txt", "w+");
  // for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
  //   auto [x, y] = wrapper.nodeLocation(nodeIdx);
  //   fprintf(fp, "%f %f\n", x, y);
  // }
  // fclose(fp);

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