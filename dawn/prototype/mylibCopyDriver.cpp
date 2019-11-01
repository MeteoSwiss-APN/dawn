#include "mylib_interface.hpp"
#include <fstream>
#include <gridtools/clang_dsl.hpp>

#include "generated_copyEdgeToCell.hpp"

using namespace mylibInterface

    int
    main() {
  int w = 20;
  Mesh m{w, w, true};
  mylib::FaceData<double> out(m);
  mylib::EdgeData<double> in(m);

  for(int i = 0; i < 10; ++i) {
    std::ofstream of("of_" + std::to_string(i) + ".vtk");
    toVtk(m, of);
    toVtk("temperature", out, m, of);

    dawn_generated::cxxnaiveico::generated<mylibInterface::mylibTag>(m, 0, in, out).run();
  }
}
