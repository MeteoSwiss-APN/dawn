#define GRIDTOOLS_CLANG_GENERATED 1 // this is important, defines meta_data_t

#include "gridtools/clang/domain.hpp"
#include "gridtools/clang_dsl.hpp"

using namespace gridtools::clang;

#include "laplacian_stencil_cxx_naive.cpp"
//#include "laplacian_stencil_from_python.cpp"
//#include "cpp/laplacian_stencil_from_standalone.cpp"

#include <fstream>

#include <tuple>
#include <vector>

using position_vector = std::vector<std::tuple<double, double>>;

// write gridtools storage as vtk
void to_VTK(const std::string& fname, const position_vector& positions, const storage_ijk_t& field,
            int dimension) {
  auto field_v = make_host_view(field);
  auto at = [dimension](position_vector pos, int i, int j) { return pos[i * dimension + j]; };

  // vtk header
  std::ofstream os;
  os.open(fname, std::ofstream::out);
  os << "# vtk DataFile Version 3.0\n2D scalar data\nASCII\nDATASET "
        "STRUCTURED_GRID\n";
  os << "DIMENSIONS " << dimension << " " << dimension << " " << 1 << "\n";

  // write strucutred mesh geometry (fun val = height)
  os << "POINTS " << dimension * dimension << " "
     << "float\n";
  for(int i = 0; i < dimension; i++) {
    for(int j = 0; j < dimension; j++) {
      auto pos = at(positions, i, j);
      os << std::get<0>(pos) << " " << std::get<1>(pos) << " " << field_v(i, j, 0) << "\n";
    }
  }

  // write height again as point data for nice coloring
  os << "POINT_DATA " << dimension * dimension << "\n";
  os << "SCALARS height float 1\n";
  os << "LOOKUP_TABLE default\n";
  for(int i = 0; i < dimension; i++) {
    for(int j = 0; j < dimension; j++) {
      os << field_v(i, j, 0) << "\n";
    }
  }

  os.close();
}

int main() {
  // grid size per dimension (complete size = N*N)
  const int N = 30;
  domain dom(N, N, 1);

  // in and output field
  meta_data_ijk_t meta_data(dom.isize(), dom.jsize(), 1);
  storage_ijk_t in(meta_data, "in"), out(meta_data, "out");

  // domain size
  double L = 10;
  // domain offset
  double low = L / 2;
  // grid spacing
  double dx = L / (N - 1);
  // genereate positions for convenience
  position_vector positions;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      positions.push_back({i * dx, j * dx});
    }
  }

  // position lookup convenience function
  auto at = [N](position_vector pos, int i, int j) { return pos[i * N + j]; };

  // populate in field
  auto in_v = make_host_view(in);
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < N; j++) {
      auto pos = at(positions, i, j);
      in_v(i, j, 0) = sin(std::get<0>(pos)) * sin(std::get<1>(pos));
    }
  }

  // visualize in field
  to_VTK("in.vtk", positions, in, N);

  // perform laplacian using gtclang generated code
  dawn_generated::cxxnaive::laplacian_stencil laplacian_naive(dom, out, in);
  laplacian_naive.set_dx(dx);
  laplacian_naive.run(out, in);

  // visualize result
  to_VTK("out.vtk", positions, out, N);

  return 0;
}
