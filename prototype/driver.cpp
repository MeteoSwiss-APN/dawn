#include <fstream>
#include <gridtools/clang_dsl.hpp>

#include "generated.hpp"
#include "my_interface.hpp"

int main() {
  int w = 20;
  mylibInterface::Mesh m{w, w, true};
  mylibInterface::Field<double> in(m), out(m);

  for(auto& f : m.faces()) {
    auto center_x = w / 2.f - (1.f / 3) * (f.vertex(0).x() + f.vertex(1).x() + f.vertex(2).x());
    auto center_y = w / 2.f - (1.f / 3) * (f.vertex(0).y() + f.vertex(1).y() + f.vertex(2).y());
    in(f) = (center_x * center_x + center_y * center_y > w / 3.) ? 1 : 0;
  }

  for(int i = 0; i < 1000; ++i) {
    std::ofstream of("of_" + std::to_string(i) + ".vtk");
    toVtk(m, of);
    toVtk("temperature", in, m, of);

    dawn_generated::cxxnaiveico::generated(mylibInterface::mylibTag{}, m, in, out).run();

    using std::swap;
    swap(in, out);
  }
}
