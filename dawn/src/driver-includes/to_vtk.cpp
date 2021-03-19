#include "to_vtk.h"

#include "cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <utility>

MeshInfoVtk mesh_info_vtk;

static const std::string fname_pre = "dsl_fields_";

class StencilFieldsVtkOutput {
  std::fstream fs;
  std::stringstream cells_ss, points_ss;
  bool wroteCellData = false, wrotePointData = false;

  int num_k_;

public:
  StencilFieldsVtkOutput(int num_k, std::string stencil_name, int iteration) : num_k_(num_k) {

    if(!mesh_info_vtk.isInitialized()) {
      throw std::runtime_error("Uninitialized vtk mesh data.");
    }

    const int num_cells = mesh_info_vtk.mesh_num_cells;
    const int num_verts = mesh_info_vtk.mesh_num_verts;

    const int* cells_vertex_idx = mesh_info_vtk.mesh_cells_vertex_idx;

    const double* vlat = mesh_info_vtk.mesh_vlat;
    const double* vlon = mesh_info_vtk.mesh_vlon;

    const double lat_range =
        *std::max_element(vlat, vlat + num_verts) - *std::min_element(vlat, vlat + num_verts);
    const double lon_range =
        *std::max_element(vlon, vlon + num_verts) - *std::min_element(vlon, vlon + num_verts);
    const double range = std::max(lat_range, lon_range);

    fs.open(fname_pre + stencil_name + "_" + std::to_string(iteration) + ".vtk", std::fstream::out);

    fs << "# vtk DataFile Version 3.0\n2D scalar data\nASCII\nDATASET "
          "UNSTRUCTURED_GRID\n";

    fs << "POINTS " << num_verts * num_k << " float\n";

    for(int k = 0; k < num_k; k++) {
      for(int nodeIter = 0; nodeIter < num_verts; nodeIter++) {
        double x = vlat[nodeIter];
        double y = vlon[nodeIter];
        double z = k / ((double)num_k) * range;
        fs << x << " " << y << " " << z << "\n";
      }
    }

    fs << "CELLS " << num_cells * num_k << " " << 4 * num_cells * num_k << "\n";

    for(int k = 0; k < num_k; k++) {
      for(int cellIter = 0; cellIter < num_cells; cellIter++) {
        fs << "3 " << cells_vertex_idx[0 * num_cells + cellIter] + k * num_verts << " "
           << cells_vertex_idx[1 * num_cells + cellIter] + k * num_verts << " "
           << cells_vertex_idx[2 * num_cells + cellIter] + k * num_verts << '\n';
      }
    }

    fs << "CELL_TYPES " << num_cells * num_k << '\n';
    for(int k = 0; k < num_k; k++) {
      for(int cellIter = 0; cellIter < num_cells; cellIter++) {
        fs << "5\n";
      }
    }
  }

  ~StencilFieldsVtkOutput() {
    fs << cells_ss.str();
    fs << points_ss.str();
    fs.close();
  }

  StencilFieldsVtkOutput(const StencilFieldsVtkOutput&) = delete;
  StencilFieldsVtkOutput(StencilFieldsVtkOutput&&) = default;
  StencilFieldsVtkOutput& operator=(const StencilFieldsVtkOutput&) = delete;
  StencilFieldsVtkOutput& operator=(StencilFieldsVtkOutput&&) = default;

  std::stringstream& cellData() {
    if(!wroteCellData) {
      cells_ss << "CELL_DATA " << mesh_info_vtk.mesh_num_cells * num_k_ << '\n';
    }
    wroteCellData = true;
    return cells_ss;
  }
  std::stringstream& pointData() {
    if(!wrotePointData) {
      points_ss << "POINT_DATA " << mesh_info_vtk.mesh_num_verts * num_k_ << '\n';
    }
    wrotePointData = true;
    return points_ss;
  }
};

// (stencil_name, iteration) -> vtk_output_handle
std::map<std::pair<std::string, int>, StencilFieldsVtkOutput> stencil_to_output_map;

static std::string formatNaNs(const double value) {
  if(std::isnan(value)) {
    return "nan";
  }

  return std::to_string(value);
}

namespace {
StencilFieldsVtkOutput& getStencilFieldsVtkOutput(int num_k, std::string stencil_name, int iter) {
  if(stencil_to_output_map.count(std::make_pair(stencil_name, iter)) == 0) {
    stencil_to_output_map.emplace(std::make_pair(stencil_name, iter),
                                  StencilFieldsVtkOutput(num_k, stencil_name, iter));
  }
  return stencil_to_output_map.at(std::make_pair(stencil_name, iter));
}

void flushAtIter(std::string stencil_name, int iter) {
  stencil_to_output_map.erase(std::make_pair(stencil_name, iter - 1));
}

double* fieldFromGpu(const double* field_gpu, const int size) {
  double* field_cpu = new double[size];
  gpuErrchk(cudaMemcpy(field_cpu, field_gpu, sizeof(double) * size, cudaMemcpyDeviceToHost));
  return field_cpu;
}

void dense_cells_to_csv(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {
  std::fstream fs;
  fs.open(std::string(stencil_name) + "_" + std::string(field_name) + "_" + std::to_string(iter) +
              ".csv",
          std::fstream::out);

  fs << "\"";
  fs << field_name;
  fs << "\"\n";

  fs << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

  for(int k = 0; k < num_k; k++) {
    for(int cellIter = 0; cellIter < mesh_info_vtk.mesh_num_cells; cellIter++) {
      if(cellIter < start_idx || cellIter > end_idx) {
        fs << 0.0 << "\n";
      } else {
        fs << field[k * dense_stride + cellIter] << "\n";
      }
    }
  }

  fs.close();
}

void dense_cells_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {

  auto& output = getStencilFieldsVtkOutput(num_k, std::string(stencil_name), iter);

  output.cellData() << "SCALARS " << std::string(field_name) << " float 1\nLOOKUP_TABLE default\n";

  for(int k = 0; k < num_k; k++) {
    for(int cellIter = 0; cellIter < mesh_info_vtk.mesh_num_cells; cellIter++) {
      if(cellIter < start_idx || cellIter > end_idx) {
        output.cellData() << 0.0 << "\n";
      } else {
        output.cellData() << formatNaNs(field[k * dense_stride + cellIter]) << "\n";
      }
    }
  }
}

void dense_verts_to_csv(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {
  std::fstream fs;
  fs.open(std::string(stencil_name) + "_" + std::string(field_name) + "_" + std::to_string(iter) +
              ".csv",
          std::fstream::out);

  fs << "\"";
  fs << field_name;
  fs << "\"\n";

  fs << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

  for(int k = 0; k < num_k; k++) {
    for(int vertIter = 0; vertIter < mesh_info_vtk.mesh_num_verts; vertIter++) {
      if(vertIter < start_idx || vertIter > end_idx) {
        fs << 0.0 << "\n";
      } else {
        fs << field[k * dense_stride + vertIter] << "\n";
      }
    }
  }

  fs.close();
}

void dense_verts_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {

  auto& output = getStencilFieldsVtkOutput(num_k, std::string(stencil_name), iter);

  output.pointData() << "SCALARS " << std::string(field_name) << " float 1\nLOOKUP_TABLE default\n";

  for(int k = 0; k < num_k; k++) {
    for(int pointIter = 0; pointIter < mesh_info_vtk.mesh_num_verts; pointIter++) {
      if(pointIter < start_idx || pointIter > end_idx) {
        output.pointData() << 0.0 << "\n";
      } else {
        output.pointData() << formatNaNs(field[k * dense_stride + pointIter]) << "\n";
      }
    }
  }
}

void dense_edges_to_csv(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {
  std::fstream fs;
  fs.open(std::string(stencil_name) + "_" + std::string(field_name) + "_" + std::to_string(iter) +
              ".csv",
          std::fstream::out);

  fs << "\"";
  fs << field_name;
  fs << "\"\n";

  fs << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

  for(int k = 0; k < num_k; k++) {
    for(int cellIter = 0; cellIter < mesh_info_vtk.mesh_num_cells; cellIter++) {
      double interpol = 0.0;
      for(int neighbor = 0; neighbor < 3; neighbor++) {
        int idx =
            mesh_info_vtk.mesh_cells_edge_idx[neighbor * mesh_info_vtk.mesh_num_cells + cellIter];
        if(idx < start_idx || idx > end_idx) {
          interpol = 0.0;
          break;
        }
        interpol += field[k * dense_stride + idx];
      }
      interpol /= double(3);
      fs << interpol << "\n";
    }
  }

  fs.close();
}

void dense_edges_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter) {

  auto& output = getStencilFieldsVtkOutput(num_k, std::string(stencil_name), iter);
  // Edges not supported by vtk, need to interpolate into cells.
  output.cellData() << "SCALARS " << std::string(field_name) << " float 1\nLOOKUP_TABLE default\n";

  for(int k = 0; k < num_k; k++) {
    for(int cellIter = 0; cellIter < mesh_info_vtk.mesh_num_cells; cellIter++) {
      double interpol = 0.0;
      for(int neighbor = 0; neighbor < 3; neighbor++) {
        int idx =
            mesh_info_vtk.mesh_cells_edge_idx[neighbor * mesh_info_vtk.mesh_num_cells + cellIter];
        if(idx < start_idx || idx > end_idx) {
          interpol = 0.0;
          break;
        }
        interpol += field[k * dense_stride + idx];
      }
      interpol /= double(3);
      output.cellData() << formatNaNs(interpol) << "\n";
    }
  }
}

} // namespace

extern "C" {

void serialize_dense_cells(int start_idx, int end_idx, int num_k, int dense_stride,
                           const double* field_gpu, const char stencil_name[50],
                           const char field_name[50], int iter) {
  const double* field = fieldFromGpu(field_gpu, dense_stride * num_k);

  dense_cells_to_csv(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);

  dense_cells_to_vtk(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);
}

void serialize_dense_verts(int start_idx, int end_idx, int num_k, int dense_stride,
                           const double* field_gpu, const char stencil_name[50],
                           const char field_name[50], int iter) {
  const double* field = fieldFromGpu(field_gpu, dense_stride * num_k);

  dense_verts_to_csv(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);

  dense_verts_to_vtk(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);
}

void serialize_dense_edges(int start_idx, int end_idx, int num_k, int dense_stride,
                           const double* field_gpu, const char stencil_name[50],
                           const char field_name[50], int iter) {
  const double* field = fieldFromGpu(field_gpu, dense_stride * num_k);

  dense_edges_to_csv(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);

  dense_edges_to_vtk(start_idx, end_idx, num_k, dense_stride, field, stencil_name, field_name,
                     iter);
}

void serialize_flush_iter(const char field_name[50], int iter) {
  flushAtIter(std::string(field_name), iter);
}
}