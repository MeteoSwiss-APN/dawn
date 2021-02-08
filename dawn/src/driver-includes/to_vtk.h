#pragma once

struct MeshInfoVtk {

  int mesh_num_edges = 0;
  int mesh_num_cells = 0;
  int mesh_num_verts = 0;
  int* mesh_cells_vertex_idx = nullptr; /*[3][num_cells]*/
  int* mesh_cells_edge_idx = nullptr;   /*[3][num_cells]*/
  double* mesh_vlon = nullptr;          /*[num_vertices]*/
  double* mesh_vlat = nullptr;          /*[num_vertices]*/

  bool isInitialized() const {
    return mesh_num_edges && mesh_num_cells && mesh_num_verts && mesh_cells_vertex_idx &&
           mesh_cells_edge_idx && mesh_vlon && mesh_vlat;
  }

  ~MeshInfoVtk() {
    if(mesh_cells_vertex_idx)
      delete[] mesh_cells_vertex_idx;
    if(mesh_cells_edge_idx)
      delete[] mesh_cells_edge_idx;
    if(mesh_vlon)
      delete[] mesh_vlon;
    if(mesh_vlat)
      delete[] mesh_vlat;
  }
};

extern MeshInfoVtk mesh_info_vtk;

extern "C" {

void dense_cells_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter);
void dense_verts_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter);
void dense_edges_to_vtk(int start_idx, int end_idx, int num_k, int dense_stride,
                        const double* field, const char stencil_name[50], const char field_name[50],
                        int iter);
}