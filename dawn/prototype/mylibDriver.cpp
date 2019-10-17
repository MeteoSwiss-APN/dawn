#include "mylib_interface.hpp"
#include <fstream>
#include <gridtools/clang_dsl.hpp>
using namespace mylibInterface;

#include "generated.hpp"

int main() {
  int w = 20;
  Mesh m{w, w, true};
  mylib::FaceData<double> out(m);
  mylib::EdgeData<double> in(m);

  // for(auto& f : m.edges()) {
  //   auto center_x = w / 2.f - (1.f / 2) * (f.vertex(0).x() + f.vertex(1).x());
  //   auto center_y = w / 2.f - (1.f / 2) * (f.vertex(0).y() + f.vertex(1).y());
  //   in[f] = (center_x * center_x + center_y * center_y > w / 3.) ? 1 : 0;
  // }

  for(int i = 0; i < 10; ++i) {
    std::ofstream of("of_" + std::to_string(i) + ".vtk");
    toVtk(m, of);
    toVtk("temperature", out, m, of);

    dawn_generated::cxxnaiveico::generated<mylibInterface::mylibTag>(m, in, out).run();
  }
}
