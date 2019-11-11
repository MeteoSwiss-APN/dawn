#include "mylib_interface.hpp"
#include <fstream>
#include <gridtools/clang_dsl.hpp>

#include "generated_Diffusion.hpp"

int main() {
  int w = 32;
  int k_size = 1;
  mylib::Grid m{w, w, true};
  mylib::FaceData<double> in(m, k_size), out(m, k_size);

  for(auto& f : m.faces()) {
    auto center_x = w / 2.f - (1.f / 3) * (f.vertex(0).x() + f.vertex(1).x() + f.vertex(2).x());
    auto center_y = w / 2.f - (1.f / 3) * (f.vertex(0).y() + f.vertex(1).y() + f.vertex(2).y());
    in(f, 0) =
        std::abs(center_x) < w / 5. && std::abs(center_y) < w / 5. && mylib::inner_face(f) ? 1 : 0;
  }

  for(int i = 0; i < 500; ++i) {
    std::ofstream of("of_" + std::to_string(i) + ".vtk");
    toVtk(m, k_size, of);
    toVtk("temperature", in, m, of);

    dawn_generated::cxxnaiveico::generated<mylibInterface::mylibTag>(m, k_size, in, out).run();

    using std::swap;
    swap(in, out);
  }
}
