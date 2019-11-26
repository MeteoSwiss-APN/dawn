#include "mylib_interface.hpp"
#include <fstream>

#include "generated_verticalSum.hpp"

int main() {
  int w = 32;
  int k_size = 5;
  mylibInterface::Mesh m{w, w, true};
  mylib::FaceData<double> out(m, k_size);
  mylib::FaceData<double> in(m, k_size);

  for(int level = 0; level < k_size; ++level) {
    for(const auto& face : m.faces()) {
      in(face, level) = 10;
    }
  }

  dawn_generated::cxxnaiveico::generated<mylibInterface::mylibTag>(m, k_size, in, out).run();

  std::ofstream of("vertSum.vtk");
  toVtk(m, out.k_size(), of);
  toVtk("temperature", out, m, of);
}