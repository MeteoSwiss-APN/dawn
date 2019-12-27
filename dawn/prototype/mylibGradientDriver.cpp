#include "mylib_interface.hpp"
#include <fstream>

#include "generated_triGradient.hpp"

int main() {
  int w = 32;
  int k_size = 1;
  mylib::Grid m{w, w, true};
  mylib::FaceData<double> faces(m, k_size);
  mylib::EdgeData<double> edges(m, k_size);

  for(auto& f : m.faces()) {
    auto center_x = (1.f / (3 * w)) * (f.vertex(0).x() + f.vertex(1).x() + f.vertex(2).x()) * M_PI;
    auto center_y = (1.f / (3 * w)) * (f.vertex(0).y() + f.vertex(1).y() + f.vertex(2).y()) * M_PI;
    faces(f, 0) = sin(center_x) * sin(center_y);
  }

  {
    std::ofstream of("signal.vtk");
    toVtk(m, k_size, of);
    toVtk("signal", faces, m, of);
  }

  dawn_generated::cxxnaiveico::gradient<mylibInterface::mylibTag>(m, k_size, faces, edges).run();

  {
    std::ofstream of("gradient.vtk");
    toVtk(m, k_size, of);
    toVtk("gradient", faces, m, of);
  }
}
