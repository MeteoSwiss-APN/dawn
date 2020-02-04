#include <fstream>
#include <vector>

#include "generated_sparseDimension.hpp"
#include "mylib_interface.hpp"

int main() {

  const int w = 10;
  const int k_size = 1;
  const int level = 0;
  const int edgesPerCell = 3;
  const int nodesPerEdge = 3;
  mylib::Grid mesh{w, w, true};
  mylib::FaceData<double> faces(mesh, k_size);
  mylib::EdgeData<double> edges(mesh, k_size);

  mylib::SparseFaceData<double> facesSparse(mesh, k_size, edgesPerCell);

  for(auto& f : mesh.faces()) {
    faces(f, level) = 0;
  }

  for(auto& e : mesh.edges()) {
    edges(e, level) = 1.;
  }

  // init sparse dimensions
  for(auto& f : mesh.faces()) {
    int jNbh = 0;
    for(auto& e : f.edges()) {
      double x0 = e->vertex(0).x();
      double y0 = e->vertex(0).y();
      double x1 = e->vertex(1).x();
      double y1 = e->vertex(1).y();
      double xm = 0.5 * (x0 + x1);
      double ym = 0.5 * (y0 + y1);
      facesSparse(f, jNbh++, level) = xm * xm + ym * ym;
    }
  }

  dawn_generated::cxxnaiveico::sparseDimension<mylibInterface::mylibTag>(mesh, k_size, faces, edges,
                                                                         facesSparse)
      .run();

  FILE* fp = fopen("sparseDimMylib.txt", "w+");
  for(auto& f : mesh.faces()) {
    double x = 0.;
    double y = 0.;
    int num_vertices = 0;
    for(auto& v : f.vertices()) {
      x += v->x();
      y += v->y();
      num_vertices++;
    }
    x /= num_vertices;
    y /= num_vertices;
    fprintf(fp, "%f %f %f\n", x, y, faces(f, level));
  }
  fclose(fp);

  // visualize in octave for a nice color gradient:
  // p = load('sparseDimMylib.txt')
  // scatter(p(:,1),p(:,2),50,p(:,3),'filled');
}
