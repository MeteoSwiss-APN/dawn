/* This is part of the netCDF package.
   Copyright 2006 University Corporation for Atmospheric Research/Unidata.
   See COPYRIGHT file for conditions of use.

   This is a very simple example which reads a 2D array of
   sample data produced by simple_xy_wr.cpp.

   This example is part of the netCDF tutorial:
   http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-tutorial

   Full documentation of the netCDF C++ API can be found at:
   http://www.unidata.ucar.edu/software/netcdf/docs/netcdf-cxx

   $Id: simple_xy_rd.cpp,v 1.5 2010/02/11 22:36:43 russ Exp $
*/

#include <assert.h>
#include <iostream>
#include <math.h>
#include <optional>
#include <string>

#include <netcdf>

#include <atlas/array/Array.h>
#include <atlas/array/ArrayView.h>
#include <atlas/array/IndexView.h>
#include <atlas/library/Library.h>
#include <atlas/library/config.h>
#include <atlas/mesh/ElementType.h>
#include <atlas/mesh/Elements.h>
#include <atlas/mesh/HybridElements.h>
#include <atlas/mesh/Mesh.h>
#include <atlas/mesh/Nodes.h>
#include <atlas/meshgenerator/detail/MeshGeneratorImpl.h>
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

std::optional<atlas::Mesh> AtlasMeshFromNetcdf(const std::string& filename) {
  atlas::Mesh mesh;

  // Open the file for read access
  try {
    netCDF::NcFile dataFile(filename.c_str(), netCDF::NcFile::read);

    // Retrieve lat/lon of vertices
    netCDF::NcVar dataLon = dataFile.getVar("vlon");
    netCDF::NcVar dataLat = dataFile.getVar("vlat");
    if(dataLon.isNull() || dataLon.isNull()) {
      std::cout << "lat or long not found!\n";
      return std::nullopt;
    }

    int lonSize = dataLon.getDim(0).getSize();
    int latSize = dataLon.getDim(0).getSize();
    if(lonSize != latSize) {
      std::cout << "lat / long not of consistent sizes!\n";
    }

    int nnodes = latSize;

    // define nodes and associated properties for Atlas meshs
    mesh.nodes().resize(nnodes);
    atlas::mesh::Nodes& nodes = mesh.nodes();
    auto xy = atlas::array::make_view<double, 2>(nodes.xy());
    auto lonlat = atlas::array::make_view<double, 2>(nodes.lonlat());

    // we currently don't care about parts, so myPart is always 0 and remotde_idx == glb_idx
    auto glb_idx_node = atlas::array::make_view<atlas::gidx_t, 1>(nodes.global_index());
    auto remote_idx = atlas::array::make_indexview<atlas::idx_t, 1>(nodes.remote_index());
    const int myPart = 0;
    auto part = atlas::array::make_view<int, 1>(nodes.partition());

    // no ghosts currently (ghost = false always) and no flags are set
    auto ghost = atlas::array::make_view<int, 1>(nodes.ghost());
    auto flags = atlas::array::make_view<int, 1>(nodes.flags());

    // read lat / lon from file
    double lonIn[lonSize];
    double latIn[latSize];
    dataLon.getVar(lonIn);
    dataLat.getVar(latIn);

    for(int nodeIdx = 0; nodeIdx < lonSize; nodeIdx++) {
      xy(nodeIdx, atlas::LON) = lonIn[nodeIdx] / (M_PI)*180;
      xy(nodeIdx, atlas::LAT) = latIn[nodeIdx] / (M_PI / 2) * 90;

      lonlat(nodeIdx, atlas::LON) = lonIn[nodeIdx] / (M_PI)*180;
      lonlat(nodeIdx, atlas::LAT) = latIn[nodeIdx] / (M_PI / 2) * 90;

      glb_idx_node(nodeIdx) = nodeIdx;
      remote_idx(nodeIdx) = nodeIdx;

      part(nodeIdx) = myPart;
      ghost(nodeIdx) = false;
      atlas::mesh::Nodes::Topology::reset(flags(nodeIdx));
    }

    // retrieve size of cell_index variable from netcdf file to determine the number of cells
    netCDF::NcVar dataCellIdx = dataFile.getVar("cell_index");
    if(dataCellIdx.isNull()) {
      std::cout << "variable cell_index not found!\n";
      return std::nullopt;
    }

    int ncells = dataCellIdx.getDim(0).getSize();

    // define cells and associated properties
    mesh.cells().add(new atlas::mesh::temporary::Triangle(), ncells);
    int tri_begin = mesh.cells().elements(0).begin();
    auto cells_part = atlas::array::make_view<int, 1>(mesh.cells().partition());
    atlas::mesh::HybridElements::Connectivity& node_connectivity = mesh.cells().node_connectivity();
    atlas::array::ArrayView<atlas::gidx_t, 1> glb_idx_cell =
        atlas::array::make_view<atlas::gidx_t, 1>(mesh.cells().global_index());

    const int numVertexPerCell = 3;
    atlas::idx_t tri_nodes[3];
    int jcell = tri_begin;

    // retrieve cell to node connectivity from netcdf file
    netCDF::NcVar dataCellNbh = dataFile.getVar("vertex_of_cell");
    if(dataCellNbh.isNull()) {
      std::cout << "variable vertex_of_cell not found!\n";
      return std::nullopt;
    }
    int cellToVertex[dataCellNbh.getDim(0).getSize()][dataCellNbh.getDim(1).getSize()];
    dataCellNbh.getVar(cellToVertex);

    for(int cellIdx = 0; cellIdx < ncells; cellIdx++) {
      tri_nodes[0] = cellToVertex[0][cellIdx] - 1;
      tri_nodes[1] = cellToVertex[1][cellIdx] - 1;
      tri_nodes[2] = cellToVertex[2][cellIdx] - 1;
      node_connectivity.set(jcell, tri_nodes);
      glb_idx_cell[cellIdx] = jcell;
      cells_part(jcell) = myPart;
      jcell++;
    }

    return mesh;
  } catch(netCDF::exceptions::NcException& e) {
    std::cout << e.what() << "\n";
    return std::nullopt;
  }
}

void debugDump(const atlas::Mesh& mesh) {
  auto lonlat = atlas::array::make_view<double, 2>(mesh.nodes().lonlat());
  const atlas::mesh::HybridElements::Connectivity& node_connectivity =
      mesh.cells().node_connectivity();

  {
    FILE* fp = fopen("netcdfMeshT.txt", "w+");
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      int nodeIdx0 = node_connectivity(cellIdx, 0);
      int nodeIdx1 = node_connectivity(cellIdx, 1);
      int nodeIdx2 = node_connectivity(cellIdx, 2);
      fprintf(fp, "%d %d %d\n", nodeIdx0 + 1, nodeIdx1 + 1, nodeIdx2 + 1);
    }
    fclose(fp);
  }

  double latmin = std::numeric_limits<double>::max();
  double latmax = -std::numeric_limits<double>::max();
  double lonmin = std::numeric_limits<double>::max();
  double lonmax = -std::numeric_limits<double>::max();

  {
    FILE* fp = fopen("netcdfMeshP.txt", "w+");
    for(int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      double lat = lonlat(nodeIdx, atlas::LAT);
      double lon = lonlat(nodeIdx, atlas::LON);

      latmin = fmin(latmin, lat);
      latmax = fmax(latmax, lat);
      lonmin = fmin(lonmin, lon);
      lonmax = fmax(lonmax, lon);

      const double R = 6371;
      double x = R * cos(lat) * cos(lon);
      double y = R * cos(lat) * sin(lon);
      double z = R * sin(lat);
      fprintf(fp, "%f %f %f\n", x, y, z);
    }
    fclose(fp);
  }

  printf("lat (%f %f) lon(%f %f)\n", latmin, latmax, lonmin, lonmax);
}

int main() {
  // Open the file for read access
  auto mesh = AtlasMeshFromNetcdf("icon_grid_0010_R02B04_G.nc");
  assert(mesh.has_value());

  debugDump(mesh.value());

  atlas::output::Gmsh gmsh("earth.msh", atlas::util::Config("coordinates", "xyz"));
  gmsh.write(mesh.value());

  atlas::Library::instance().finalise();

  printf("done!\n");
  return 0;
}
