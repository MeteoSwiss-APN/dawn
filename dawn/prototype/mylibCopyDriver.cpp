#include "mylib_interface.hpp"
#include <fstream>
#include <gridtools/clang_dsl.hpp>

#include "generated_copyEdgeToCell.hpp"

int main() {
  int w = 10;
  int k_size = 10;
  mylibInterface::Mesh m{w, w, false};
  mylib::FaceData<double> out(m, k_size);
  mylib::EdgeData<double> in(m, k_size);

  for(int i = 0; i < 10; ++i) {
    std::ofstream of("of_" + std::to_string(i) + ".vtk");
    toVtk(m, out.k_size(), of);
    toVtk("temperature", out, m, of);

    dawn_generated::cxxnaiveico::generated<mylibInterface::mylibTag>(m, k_size, in, out).run();
  }
}
