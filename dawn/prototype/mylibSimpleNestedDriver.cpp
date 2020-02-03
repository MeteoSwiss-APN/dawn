#include <assert.h>
#include <fstream>
#include <vector>

#include "generated_NestedSimple.hpp"
#include "mylib_interface.hpp"

int main() {
  const int w = 10;
  const int k_size = 1;
  const int level = 0;
  mylib::Grid mesh{w, w, true};
  mylib::FaceData<double> faces(mesh, k_size);
  mylib::EdgeData<double> edges(mesh, k_size);
  mylib::VertexData<double> nodes(mesh, k_size);

  for(auto& f : mesh.faces()) {
    faces(f, level) = 0;
  }

  for(auto& n : mesh.vertices()) {
    nodes(n, level) = 1.;
  }

  dawn_generated::cxxnaiveico::nested<mylibInterface::mylibTag>(mesh, k_size, faces, edges, nodes)
      .run();

  // each vertex stores value 1                 1
  // vertices are reduced onto edges            2
  // each face reduces its edges (3 per face)   6
  for(auto& f : mesh.faces()) {
    assert(fabs(faces(f, level) - 6) < 1e-12);
  }
}
